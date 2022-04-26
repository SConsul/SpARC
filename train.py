import os
import json
import wandb
import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from evaluate import evaluate
from utils.datasets.dataset import QADataset
from utils.datasets.dataset_sim import QAPairsDataset
from utils.loss.loss import l1_loss
from utils.loss.sim_loss import build_sim_loss

os.environ["TOKENIZERS_PARALLELISM"] = "true"  # is this ok?


def passed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', default=None, type=str, help="Wandb project name")
    parser.add_argument('--train_path', default="./beliefbank-data-sep2021/qa_train.json")
    parser.add_argument('--val_path', default="./beliefbank-data-sep2021/constraints_qa.json")
    parser.add_argument('--model_path', type=str, required=True, help="Dir to save results")
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_decay', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--freeze_backbone', action='store_true', default=False)
    parser.add_argument('--adapter', action='store_true', default=False)
    # Options: lm_head, encoder.final_layer_norm, etc
    parser.add_argument('--layer_names', nargs='+', type=str, default=[])
    parser.add_argument('--l1_reg', type=float, default=None)
    parser.add_argument('--sim', type=float, default=None)
    parser.add_argument('--ce_loss', type=float, default=1.0)
    parser.add_argument('--token_type', type=str, default=None)
    parser.add_argument('--sim_type', type=str, default='batch', choices=['batch', 'angle', 'moco'])
    args = parser.parse_args()
    print(args)
    return args


def check_register_hook(name, config_layer_names):
    name_parts = name.split('.')
    exact_match = name in config_layer_names

    all_match = 'all' in config_layer_names
    enc_match = (('enc' in config_layer_names) or all_match) and \
                (name_parts[0] == 'encoder') and \
                (name_parts[-1] in ['layer_norm', 'final_layer_norm'])
    dec_match = (('dec' in config_layer_names) or all_match) and \
                (name_parts[0] == 'decoder') and \
                (name_parts[-1] in ['layer_norm', 'final_layer_norm'])

    adp_all_match = 'adapter_all' in config_layer_names
    adp_enc_match = (('adapter_enc' in config_layer_names) or adp_all_match) and \
                    (name_parts[0] == 'encoder') and \
                    (name_parts[-1] in ['adapter_up'])
    adp_dec_match = (('adapter_dec' in config_layer_names) or adp_all_match) and \
                    (name_parts[0] == 'decoder') and \
                    (name_parts[-1] in ['adapter_up'])
    return exact_match or enc_match or dec_match or adp_enc_match or adp_dec_match


def register_hooks(model, config, activation):
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output

        return hook

    names = []
    for name, layer in model.named_modules():
        if check_register_hook(name, config['layer_names']):
            print(f"Register hook on {name}")
            layer.register_forward_hook(get_activation(name))
            names.append(name)

    return names


def train(model, tokenizer, train_dataset, val_dataset, writer, config):
    if config['adapter']:
        # optim_groups = [p for n, p in model.named_parameters()
        #                 if len(n.split('.')) > 5 and n.split('.')[5] == 'adapters']
        model.train_adapter("beliefbank")

        optim_groups = model.parameters()
    else:
        no_decay = ["bias", "LayerNorm.weight"]
        param_gen = model.lm_head if config['freeze_backbone'] else model

        params_decay = [p for n, p in param_gen.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in param_gen.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": config['weight_decay']},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        model.train()

    optimizer = optim.AdamW(optim_groups, lr=config['learning_rate'], betas=config['betas'])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                  num_workers=config['num_workers'], shuffle=True, drop_last=True)

    activation = {}
    model_layer_names = []
    if config['l1_reg'] is not None or config['sim'] is not None:
        print(f"L1 sparsity on {config['layer_names']}")
        model_layer_names = register_hooks(model, config, activation)

    if config['sim'] is not None:
        src_len, tgt_len = train_dataset.get_activation_src_tgt_len()
        sim_loss_fn = build_sim_loss(config['sim_type'], model_layer_names, src_len, tgt_len)
        sim_loss_fn.to(config['device'])

    it_n = 0
    for epoch in range(config['max_epochs']):
        losses = []
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for it, (x, a, y, token_ids) in pbar:
            x = x.to(config['device'])  # (b, 1 or 2, InL)
            a = a.to(config['device'])  # (b, 1 or 2, InL)
            y = y.to(config['device'])  # (b, 1 or 2, OutL)
            token_ids = token_ids.to(config['device'])  # (b, 1 or 2, I)
            b, s, inL = x.shape
            _, _, outL = y.shape

            # Collapse batch dimension so model gets (b*s, L) shape tensors
            out = model(input_ids=x.view(-1, inL), attention_mask=a.view(-1, inL), labels=y.view(-1, outL))
            ce_loss = out.loss

            ce_loss = config['ce_loss'] * ce_loss.mean()  # collapse all losses if they are scattered on multiple gpus

            l1_reg_loss = torch.tensor(0.0, device=config['device'])
            if config['l1_reg'] is not None:
                for name in activation:
                    l1_reg_loss += config['l1_reg'] * l1_loss(activation[name], token_ids.view(b*s, -1))

            sim_loss = torch.tensor(0.0, device=config['device'])
            if config['sim'] is not None:
                for name in activation:
                    sim_loss += config['sim'] * sim_loss_fn(activation[name], token_ids.view(b*s, -1), name)

            loss = ce_loss + l1_reg_loss + sim_loss
            model.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm_clip'])
            optimizer.step()

            # report progress
            pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}")

            losses.append((ce_loss.item(), l1_reg_loss.item(), sim_loss.item()))

            if (it % 100) == 0:
                step_metrics = {
                    "Train/CELoss-Iter": ce_loss.item(), "Train/L1Loss-Iter": l1_reg_loss.item(),
                    "Train/SimLoss-Iter": sim_loss.item(), "Train/Loss-Iter": loss.item(),
                    'Train/grad-norm': grad_norm.item(), "Train/step": it_n + 1
                }
                for name, val in step_metrics.items():
                    writer.add_scalar(name, val, it_n + 1)

                if config['wandb']:
                    wandb.log(step_metrics)

                it_n += 100

        # Log average loss over epoch
        losses = torch.as_tensor(losses)
        mean_ce, mean_l1, mean_sim = losses.mean(dim=0)
        epoch_metrics = {
            "TrainE/CELoss-Epoch": mean_ce, "TrainE/L1Loss-Epoch": mean_l1, "TrainE/SimLoss-Epoch": mean_sim,
            "TrainE/Loss-Epoch": mean_ce + mean_l1 + mean_sim, "TrainE/step": epoch+1,
        }
        for name, val in epoch_metrics.items():
            writer.add_scalar(name, val, epoch + 1)

        if config['wandb']:
            wandb.log(epoch_metrics)

        # Evaluate on validation set
        singlehop_path = os.path.join(config['val_path'], f'singlehop_{epoch+1}.json')
        multihop_path = os.path.join(config['val_path'], f'multihop_{epoch+1}.json')
        f1, consis = evaluate(model, tokenizer, val_dataset, config['val_batch_size'], config['device'],
                              singlehop_path, multihop_path)
        if config['wandb']:
            wandb.log({"Val/F1": f1, "Val/Consistency": consis, "Val/step": epoch+1})

        # save checkpoint
        if ((epoch + 1) % 5) == 0:
            model_path = os.path.join(config['model_path'], f"{epoch + 1}.bin")
            torch.save(model.state_dict(), model_path)

    writer.flush()


def main():
    args = passed_arguments()
    # Set up wandb
    if args.wandb is not None:
        with open('wandb.json', 'r') as f:
            login_key = json.load(f)['login']
        wandb.login(key=login_key)
        wandb.init(project=args.wandb, entity="team-sparc")
        wandb.config = args.__dict__

    os.makedirs(args.model_path, exist_ok=True)

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("allenai/macaw-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-large")
    if args.adapter:
        model.add_adapter("beliefbank", config="pfeiffer")
        model.set_active_adapters("beliefbank")
    model = model.to(device)
    # model = torch.nn.DataParallel(model).to(device)
    if args.wandb is not None:
        wandb.watch(model)

    if args.sim is not None:
        train_dataset = QAPairsDataset(args.train_path, tokenizer, token_type=args.token_type)
    else:
        train_dataset = QADataset(args.train_path, tokenizer)

    val_dataset = QADataset(args.val_path, tokenizer)
    val_path = os.path.join(args.model_path, 'val_results')
    os.makedirs(val_path, exist_ok=True)

    logdir = os.path.join(args.model_path, 'logs')
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    config = {
        'wandb': args.wandb is not None,
        'device': device,
        'max_epochs': args.max_epochs,
        'batch_size': args.batch_size,
        'val_batch_size': args.val_batch_size,
        'learning_rate': args.lr,
        'betas': (0.9, 0.95),
        'grad_norm_clip': 1.0,
        'weight_decay': args.weight_decay,  # only applied on matmul weights
        'l1_reg': args.l1_reg,
        'freeze_backbone': args.freeze_backbone,
        # learning rate decay params: linear warmup followed by cosine decay to 10% of original
        'lr_decay': args.lr_decay,
        # checkpoint settings
        'model_path': args.model_path,
        'val_path': val_path,
        'num_workers': args.num_workers,  # for DataLoader
        'adapter': args.adapter,
        'layer_names': args.layer_names,
        'sim': args.sim,
        'ce_loss': args.ce_loss,
        'token_type': args.token_type,
        'sim_type': args.sim_type
    }
    train(model, tokenizer, train_dataset, val_dataset, writer, config)


if __name__ == "__main__":
    main()
