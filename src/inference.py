import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data.dataloader import DataLoader

from utils.dataset import QADataset
from utils.preprocess_utils import DataRow


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def infer(model, test_dataset, qa, batch_size, device):

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    pbar = tqdm(zip(test_dataloader, batch(qa, batch_size)), total=len(test_dataloader))

    output_preds = []
    for inputs, data_text in pbar:
        q_ids, attn, _, _ = inputs
        q_ids = q_ids.to(device)
        attn = attn.to(device)

        output_sequences = model.generate(
            input_ids=q_ids.squeeze(1),  # (b, 1, L) -> (b, L)
            attention_mask=attn.squeeze(1),  # (b, 1, L) -> (b, L)
            do_sample=False,  # disable sampling to test if batching affects output
        )

        preds = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        # q_s = tokenizer.batch_decode(q, skip_special_tokens=True)
        # y[y==-100] = tokenizer.pad_token_id
        # a_s = tokenizer.batch_decode(y, skip_special_tokens=True)
        for pred, data_row in zip(preds, data_text):
            # Out format is "$answer$ = Yes"
            pred_text = pred.split("=")[1].strip().lowercase()
            output_preds.append(
                DataRow(question=data_row['question'], answer=data_row['answer'],
                        source=data_row['source'], target=data_row['target'],
                        gold=False, id=data_row['id'], link_type=data_row['link_type'],
                        pred=pred_text)
            )

    return output_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', default="./beliefbank-data-sep2021/qa_val.json")
    parser.add_argument('--out_path', default="./beliefbank-data-sep2021/baseline.json")
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--adapter', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained("allenai/macaw-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-large")
    if args.adapter:
        model.add_adapter("beliefbank", config="pfeiffer")
        model.set_active_adapters("beliefbank")

    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model = model.to(device)

    with open(args.in_path, 'r') as f:
        qa = json.load(f)

    test_dataset = QADataset(args.in_path, tokenizer)

    output_preds = infer(model, test_dataset, qa, args.batch_size, device)

    output_preds = [dr._asdict() for dr in output_preds]
    with open(args.out_path, 'w') as outfile:
        json.dump(output_preds, outfile, indent=1)
