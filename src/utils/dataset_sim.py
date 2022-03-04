import json
import torch
from torch.utils.data import Dataset


class QAPairsDataset(Dataset):
    def __init__(self, json_filepath, tokenizer, max_source_len=64, max_target_len=8):
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        self.data = data
        self.tokenizer = tokenizer

        def gen_q(question):
            return '$answer$ ; $question$ = ' + question + ' ; $mcoptions$ = (A) Yes (B) No'

        def gen_a(answer):
            return "$answer$ = " + answer

        self.input_strings = [(gen_q(line1['question']), gen_q(line2['question']))
                              for line1, line2 in self.data]
        self.output_strings = [(gen_a(line1['answer']), gen_a(line2['answer']))
                               for line1, line2 in self.data]
        self.source_len = max_source_len
        self.target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q1, q2 = self.input_strings[idx]
        a1, a2 = self.output_strings[idx]
        inp1 = self.tokenizer(q1, return_tensors='pt', max_length=self.source_len,
                              padding="max_length", truncation=True)
        inp2 = self.tokenizer(q2, return_tensors='pt', max_length=self.source_len,
                              padding="max_length", truncation=True)
        in_token_ids = torch.cat((inp1.input_ids, inp2.input_ids), dim=0)  # (2, InL)
        in_attn_mask = torch.cat((inp1.attention_mask, inp2.attention_mask), dim=0)  # (2, InL)

        out1 = self.tokenizer(a1, return_tensors='pt', max_length=self.target_len,
                              padding="max_length", truncation=True)
        out2 = self.tokenizer(a2, return_tensors='pt', max_length=self.target_len,
                              padding="max_length", truncation=True)
        out_token_ids = torch.cat((out1.input_ids, out2.input_ids), dim=0)  # (2, OutL)

        # replace padding id's of labels by -100 for CrossEntropy to ignore (-100 is ignore index)
        out_token_ids[out_token_ids == self.tokenizer.pad_token_id] = -100

        # Remove batch dimension (as PyTorch will handle batching)
        # (2, InL), (2, InL), (2, OutL)
        return in_token_ids, in_attn_mask, out_token_ids
