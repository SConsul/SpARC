import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import train
from torch.utils.data import Dataset

class tokenedDataset(Dataset):
    def __init__(self, json_filepath, tokenizer):
        self.data = json.load(open(json_filepath))
        self.input_strings = ["$question$ = "+ line['question'] for line in self.data]
        self.output_strings = ["$answer$ = " + line['answer'] for line in self.data]

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return (self.input_strings[idx], self.output_strings[idx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default="/beliefbank-data-sep2021/qa.json")
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_decay', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--model_path', default="runs/baseline")
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("allenai/macaw-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-large")
    model = torch.nn.DataParallel(model).to(device)

    train_dataset = tokenedDataset(args.train_path)

    config = {
        'device': device,
        'max_epochs': args.max_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'betas': (0.9, 0.95),
        'grad_norm_clip': 1.0,
        'weight_decay':  args.weight_decay, # only applied on matmul weights
        # learning rate decay params: linear warmup followed by cosine decay to 10% of original
        'lr_decay': args.lr_decay,
        # checkpoint settings
        'chkpt_path': args.model_path,
        'num_workers': args.num_workers # for DataLoader
    }
    train(model, tokenizer, train_dataset, config)

if __name__=="__main__":
    main()