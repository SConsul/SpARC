
import torch
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdapterType
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from bertviz import head_view, model_view

tokenizer = AutoTokenizer.from_pretrained("allenai/macaw-large")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-large",output_attentions=True )

q1 = '$answer$ ; $mcoptions$ = (A) Yes (B) No ; $question$ = ' + 'Is a poodle a dog?'
a1 = "$answer$ = Yes"

input = tokenizer(q1, return_tensors='pt', add_special_tokens=True)
out = tokenizer(a1, return_tensors='pt',add_special_tokens=True)
op = model(input_ids=input.input_ids,attention_mask=input.attention_mask, decoder_input_ids=out.input_ids)
encoder_text = tokenizer.convert_ids_to_tokens(input.input_ids[0])
decoder_text = tokenizer.convert_ids_to_tokens(out.input_ids[0])

model_view(op[-1],encoder_text)