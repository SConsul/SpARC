import json
import torch
from torch.utils.data import Dataset


class QADataset(Dataset):
    def __init__(self, json_filepath, tokenizer, max_source_len=64, max_target_len=8):
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        self.data = data
        self.tokenizer = tokenizer

        self.input_strings = ["$answer$ ; $question$ = " + line['question']
                              + ' ; $mcoptions$ = (A) Yes (B) No' for line in self.data]
        self.output_strings = ["$answer$ = " + line['answer'] for line in self.data]
        self.source_len = max_source_len
        self.target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp = self.tokenizer(
            self.input_strings[idx], return_tensors='pt',
            max_length=self.source_len, padding="max_length", truncation=True
        )
        in_token_ids = inp.input_ids
        in_attn_mask = inp.attention_mask

        out = self.tokenizer(
            self.output_strings[idx], return_tensors='pt',
            max_length=self.target_len, padding="max_length", truncation=True
        )
        out_token_ids = out.input_ids

        # replace padding id's of labels by -100 for CrossEntropy to ignore (-100 is ignore index)
        out_token_ids[out_token_ids == self.tokenizer.pad_token_id] = -100

        # Keep batch dimension (as PyTorch will handle batching)
        # Shapes (1, InL), (1, InL), (1, OutL)
        return in_token_ids, in_attn_mask, out_token_ids, torch.tensor([[-1]], dtype=torch.long)
