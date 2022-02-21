import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

class tokenedDataset(Dataset):
    def __init__(self, json_filepath,tokenizer):
        self.data = json.load(open(json_filepath))

        self.input_strings = ["$answer$ ; $mcoptions$ ; $question$ = "+ line['question'] for line in self.data]
        self.output_strings = ["$answer$ = " + line['answer'] for line in self.data]
        self.source_len = 100#max(len(s.split()) for s in self.input_strings)
        self.target_len = 100#max(len(t.split()) for t in self.output_strings)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        src_text = str(self.input_strings[idx])
        src_text = " ".join(src_text.split())
        tgt_text = str(self.output_strings[idx])
        tgt_text = " ".join(tgt_text.split())
        return (self.tokenizer.batch_encode_plus([src_text],
                                            return_tensors="pt",
                                            max_length=self.source_len,
                                            pad_to_max_length=True,
                                            padding="max_length",
                                            truncation=True), 
                tgt_text)                            
                # self.tokenizer.batch_encode_plus([tgt_text],
                #                             return_tensors="pt",
                #                             max_length=self.source_len,
                #                             pad_to_max_length=True,
                #                             padding="max_length",
                #                             truncation=True)
                # )

def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', default="./beliefbank-data-sep2021/qa.json")
    parser.add_argument('--out_path', default="./beliefbank-data-sep2021/baseline.json")
    args = parser.parse_args()

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    output_preds = []
    tokenizer = AutoTokenizer.from_pretrained("allenai/macaw-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-large")
    model = model.to(device)
    test_dataset = tokenedDataset(args.in_path,tokenizer)

    pbar = tqdm(enumerate(test_dataset), total=len(test_dataset))
    for id, (q,a_gt) in pbar:
        # forward the model
        outputs = model.generate(input_ids = q.input_ids.to(device))
        preds = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        data = {'id':id,
                'q':tokenizer.decode(q.input_ids[0], skip_special_tokens=True),
                'pred': preds,
                 'tgt': a_gt   
                }
        output_preds.append(data)

        count+=1
        
    with open(args.out_path, 'w') as outfile:
        json.dump(output_preds, outfile, indent=4)
    outfile.close()

if __name__ =="__main__":
    infer()