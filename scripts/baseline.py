import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from train import train
from utils.dataset import QADataset


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
    args = parser.parse_args()
    return args


def main():
    args = passed_arguments()

    os.makedirs(args.model_path, exist_ok=True)

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("allenai/macaw-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-large")
    model = model.to(device)
    # model = torch.nn.DataParallel(model).to(device)

    train_dataset = QADataset(args.train_path, tokenizer)

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
        'adapter':args.adapter
    }
    train(model, train_dataset, config)


if __name__ == "__main__":
    main()
