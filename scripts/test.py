import torch
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

class tokenedDataset(Dataset):
    def __init__(self, json_filepath,tokenizer):
        self.data = json.load(open(json_filepath))

        self.input_strings = ["$question$ = "+ line['question']+"; $answer$ = " for line in self.data]
        self.output_strings = ["$answer$ = " + line['answer'] for line in self.data]
        self.source_len = max(len(s.split()) for s in self.input_strings)
        self.target_len = max(len(t.split()) for t in self.output_strings)
        print("src_len", self.source_len)
        print("tgt_len", self.target_len)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        src_text = str(self.input_strings[idx])
        src_text = " ".join(src_text.split())
        print("src_text=",src_text)
        tgt_text = str(self.output_strings[idx])
        tgt_text = " ".join(tgt_text.split())
        print("tgt_text=",tgt_text)
        return (tokenizer.batch_encode_plus([src_text],
                                            return_tensors="pt",
                                            max_length=self.source_len,
                                            pad_to_max_length=True,
                                            padding="max_length",
                                            truncation=True), 
                tokenizer.batch_encode_plus([tgt_text],
                                            return_tensors="pt",
                                            max_length=self.source_len,
                                            pad_to_max_length=True,
                                            padding="max_length",
                                            truncation=True)
                )

def test():
    tokenizer = AutoTokenizer.from_pretrained("allenai/macaw-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-large")
    test_dataset = tokenedDataset('beliefbank-data-sep2021/qa.json',tokenizer)
    for it, (q,a_gt) in enumerate(test_dataset):
    print("Q=",q)
    print("A=",a_gt)
    # forward the model
    # model_outputs = model(input_ids=q.input_ids, attention_mask=q.attention_mask,decoder_input_ids=a_gt.input_ids)
    generated_ids = model.generate(input_ids = q.input_ids, 
                                    attention_mask=q.attention_mask,
                                    max_length=150, 
                                    num_beams=2,
                                    repetition_penalty=2.5, 
                                    length_penalty=1.0, 
                                    early_stopping=True)
    
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    print("PRED=",preds)
    # target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in a_gt]
    # # output = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
    # print("TARGET=",target)

if __name__ =="__main__":
    test();