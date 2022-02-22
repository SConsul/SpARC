import os
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader


def train(model, tokenizer, train_dataset, config):
    no_decay = ["bias", "LayerNorm.weight"]
    params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optim_groups = [
        {"params": params_decay, "weight_decay": config['weight_decay']},
        {"params": params_nodecay, "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optim_groups, lr=config['learning_rate'], betas=config['betas'])

    model.train()
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                  num_workers=config['num_workers'])
    losses=[]

    for epoch in range(config['max_epochs']):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for it, (x, a, y) in pbar:
            x = x.to(config['device'])
            a = a.to(config['device'])
            y = y.to(config['device'])

            out = model(input_ids=x, attention_mask=a, labels=y)
            loss = out.loss

            loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
            losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm_clip'])
            optimizer.step()
            
            # report progress
            pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}")

        #save checkpoint
        if (epoch % 10) == 0:
            model_path = os.path.join(config['model_path'], f"{epoch}.bin")
            torch.save(model.state_dict(), model_path)
