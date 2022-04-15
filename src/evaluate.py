import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from inference import infer
from utils.accuracy import f1_score
from utils.dataset import QADataset
from utils.beliefbank_preprocess import json_serialize, flatten
from utils.consistency import gen_belief_graph, eval_consistency


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default="./beliefbank-data-sep2021/qa_val.json",
                        help="Path to val/test file questions.")
    parser.add_argument('--out_path', type=str, required=True,
                        help="Saves single hop preds. Path to runs/experiment/qa_inf.json")
    parser.add_argument('--consistency_path', type=str, required=True,
                        help="Saves multihop preds. Path to runs/experiment/qa_consistency.json")
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

    # First get single hop preds and accuracy
    with open(args.in_path, 'r') as f:
        singlehop_qa = json.load(f)
    singlehop_dataset = QADataset(args.in_path, tokenizer)

    singlehop_preds = infer(model, singlehop_dataset, singlehop_qa, args.batch_size, device)

    singlehop_preds = [dr._asdict() for dr in singlehop_preds]
    with open(args.out_path, 'w') as outfile:
        json.dump(singlehop_preds, outfile, indent=1)

    f1, skip = f1_score(singlehop_preds)
    print(f"Accuracy: {f1}, skipped: {skip}")

    # Now get multihop questions
    adj_list, multihop_adj_list = gen_belief_graph(singlehop_preds)

    multihop_qa = flatten(json_serialize(multihop_adj_list).values())
    with open(args.consistency_path, 'w') as f:
        json.dump(multihop_qa, f, indent=1)
    multihop_dataset = QADataset(args.consistency_path, tokenizer)

    multihop_preds = infer(model, multihop_dataset, multihop_qa, args.batch_size, device)
    multihop_preds = [dr._asdict() for dr in multihop_preds]
    with open(args.consistency_path, 'w') as f:
        json.dump(multihop_qa, f, indent=1)

    consis = eval_consistency(multihop_preds)
    print(f"Consistency: {consis}")




