import tqdm
import torch
from torch import NoneType
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
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
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'],num_workers=config['num_workers'])
    losses=[]

    for epoch in range(config['max_epochs']):
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for it, (q,a_gt) in pbar:
            q = tokenizer.encode(q, return_tensors="pt")
            a_gt = tokenizer.encode(a_gt, return_tensors="pt")
            q = q.to(config['device'])
            a_gt = a_gt.to(config['device'])

            # forward the model
            loss = model(input_ids=q, labels=a_gt).loss
            loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
            losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm_clip'])
            optimizer.step()
            
            # report progress
            pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}")

        #save checkpoint
        model_path = config['chkpt_path']+"_"+str(epoch)
        torch.save(model.state_dict(),model_path)
