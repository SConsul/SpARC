import os
import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.dataset import QADataset
from utils.loss import binary_sim_loss
from utils.dataset_sim import QAPairsDataset
from torch.utils.tensorboard import SummaryWriter


def passed_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default="./beliefbank-data-sep2021/qa.json")
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_decay', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--model_path', default="runs/baseline")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--l1_reg', type=float, default=None)
    parser.add_argument('--freeze_backbone', action='store_true', default=False)
    parser.add_argument('--adapter', action='store_true', default=False)
    # Options: lm_head, encoder.final_layer_norm, etc
    parser.add_argument('--layer_names', nargs='+', type=str, default=[])
    parser.add_argument('--sim', type=float, default=None)
    parser.add_argument('--token_type', type=str, default=None)
    args = parser.parse_args()
    return args


def train(model, train_dataset, writer, config):
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
                                  num_workers=config['num_workers'])

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    l1_layers = []

    if config['l1_reg'] is not None or config['sim'] is not None:
        print(f"L1 sparsity on {config['layer_names']}")
        for name, layer in model.named_modules():
            if name in config['layer_names']:
                print(f"Register hook on {name}")
                layer.register_forward_hook(get_activation(name))
                l1_layers.append(name)

            if ('enc' in config['layer_names']) or ('all' in config['layer_names']):
                layer_name_parts = name.split('.')
                if layer_name_parts[0] == 'encoder' and layer_name_parts[-1] in ['layer_norm', 'final_layer_norm']:
                    print(f"Register hook on {name}")
                    layer.register_forward_hook(get_activation(name))
                    l1_layers.append(name)

            if ('dec' in config['layer_names']) or ('all' in config['layer_names']):
                layer_name_parts = name.split('.')
                if layer_name_parts[0] == 'decoder' and layer_name_parts[-1] in ['layer_norm', 'final_layer_norm']:
                    print(f"Register hook on {name}")
                    layer.register_forward_hook(get_activation(name))
                    l1_layers.append(name)

    it_n = 0
    for epoch in range(config['max_epochs']):
        losses = []
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for it, (x, a, y,idx) in pbar:
            x = x.to(config['device'])  # (b, 1 or 2, InL)
            a = a.to(config['device'])  # (b, 1 or 2, InL)
            y = y.to(config['device'])  # (b, 1 or 2, OutL)

            b, s, inL = x.shape
            _, _, outL = y.shape
            
            # Collapse batch dimension so model gets (b*s, L) shape tensors
            out = model(input_ids=x.view(-1, inL), attention_mask=a.view(-1, inL), labels=y.view(-1, outL))
            ce_loss = out.loss

            ce_loss = ce_loss.mean()  # collapse all losses if they are scattered on multiple gpus

            l1_reg_loss = torch.tensor(0.0, device=config['device'])
            if config['l1_reg'] is not None:
                for name in l1_layers:
                    l1_regularization = config['l1_reg'] * torch.norm(activation[name], 1)
                    l1_reg_loss += l1_regularization

            sim_loss = torch.tensor(0.0, device=config['device'])
            if config['sim'] is not None:
                for name in l1_layers:
                        sim_loss += config['sim'] * binary_sim_loss(activation[name],idx)
                        
            loss = ce_loss + l1_reg_loss + sim_loss
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm_clip'])
            optimizer.step()

            # report progress
            pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}")

            losses.append((ce_loss.item(), l1_reg_loss.item(), sim_loss.item()))

            if (it % 100) == 0:
                writer.add_scalar("Train/CELoss/Iter", ce_loss.item(), it_n+1)
                writer.add_scalar("Train/L1Loss/Iter", l1_reg_loss.item(), it_n + 1)
                writer.add_scalar("Train/SimLoss/Iter", sim_loss.item(), it_n + 1)
                writer.add_scalar("Train/Loss/Iter", loss.item(), it_n + 1)
                it_n += 100

        # Log average loss over epoch
        losses = torch.as_tensor(losses)
        mean_ce, mean_l1, mean_sim = losses.mean(dim=0)
        writer.add_scalar("Train/CELoss/Epoch", mean_ce, epoch + 1)
        writer.add_scalar("Train/L1Loss/Epoch", mean_l1, epoch + 1)
        writer.add_scalar("Train/SimLoss/Epoch", mean_sim, epoch + 1)
        writer.add_scalar("Train/Loss/Epoch", mean_ce + mean_l1 + mean_sim, epoch + 1)

        # save checkpoint
        if ((epoch + 1) % 5) == 0:
            model_path = os.path.join(config['model_path'], f"{epoch + 1}.bin")
            torch.save(model.state_dict(), model_path)

    writer.flush()


def main():
    args = passed_arguments()

    os.makedirs(args.model_path, exist_ok=True)

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("allenai/macaw-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-large")
    if args.adapter:
        model.add_adapter("beliefbank", config="pfeiffer")
        model.set_active_adapters("beliefbank")
    model = model.to(device)
    # model = torch.nn.DataParallel(model).to(device)

    if args.sim is not None:
        train_dataset = QAPairsDataset(args.train_path, tokenizer, args.token)
    else:
        train_dataset = QADataset(args.train_path, tokenizer)

    logdir = os.path.join(args.model_path, 'logs')
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    config = {
        'device': device,
        'max_epochs': args.max_epochs,
        'batch_size': args.batch_size,
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
        'num_workers': args.num_workers,  # for DataLoader
        'adapter': args.adapter,
        'layer_names': args.layer_names,
        'sim': args.sim,
        'token_type': args.token_type
    }
    train(model, train_dataset, writer, config)


if __name__ == "__main__":
    main()
