import os
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader


def train(model, train_dataset, config):
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
    losses = []
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook
    
    l1_layers = []

    if config['l1_reg'] is not None:
        print(f"L1 sparsity on {config['layer_names']}")
        for name, layer in model.named_modules():
            if name in config['layer_names']:
                print(f"Register hook on {name}")
                layer.register_forward_hook(get_activation(name))
                l1_layers.append(name)
            
            if config['layer_names'] == 'enc' or config['layer_names'] == 'all':
                layer_name_parts = name.split('.')
                if layer_name_parts[0] == 'encoder' and layer_name_parts[-1] in ['layer_norm', 'final_layer_norm']: 
                    print(f"Register hook on {name}")
                    layer.register_forward_hook(get_activation(name))
                    l1_layers.append(name)

            if config['layer_names'] == 'dec' or config['layer_names'] == 'all':
                layer_name_parts = name.split('.')
                if layer_name_parts[0] == 'decoder' and layer_name_parts[-1] in ['layer_norm', 'final_layer_norm']: 
                    print(f"Register hook on {name}")
                    layer.register_forward_hook(get_activation(name))
                    l1_layers.append(name)


    for epoch in range(config['max_epochs']):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for it, (x, a, y) in pbar:
            x = x.to(config['device'])
            a = a.to(config['device'])
            y = y.to(config['device'])

            out = model(input_ids=x, attention_mask=a, labels=y)
            loss = out.loss

            loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
            losses.append(loss.item())

            if config['l1_reg'] is not None:
                for name in l1_layers:
                    l1_regularization = config['l1_reg'] * torch.norm(activation[name], 1)
                    loss += l1_regularization

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm_clip'])
            optimizer.step()

            # report progress
            pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}")

        # save checkpoint
        if (epoch % 1) == 0:
            model_path = os.path.join(config['model_path'], f"{epoch + 1}.bin")
            torch.save(model.state_dict(), model_path)
