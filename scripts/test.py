import tqdm
import torch

from torch.utils.data.dataloader import DataLoader

def evaluate(model, tokenizer, test_dataset, config):
    with torch.no_grad():
    
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'],num_workers=config['num_workers'])
        corr = 0
        num = 0

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        for it, (q,a_gt) in pbar:
            q = tokenizer.encode(q, return_tensors="pt")
            a_gt = tokenizer.encode(a_gt, return_tensors="pt")
            q = q.to(config['device'])
            a_gt = a_gt.to(config['device'])

            # forward the model
            a = model.generate(q,max_length=200)
            output = tokenizer.batch_decode(a, skip_special_tokens=True)
            print(a)

            # report progress
            