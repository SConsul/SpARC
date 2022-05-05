import json
import argparse


def f1_score(results):
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    skip_count = 0
    for pred in results:
        if pred["pred"] == "yes":
            if pred["answer"] == "yes":
                TP += 1
            else:
                FP += 1
        elif pred["pred"] == "no":
            if pred["answer"] == "yes":
                FN += 1
            else:
                TN += 1
        else:
            skip_count += 1

    denom = TP + 0.5 * (FP + FN)
    return (TP / denom) if denom > 0 else 0., skip_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', default="./beliefbank-data-sep2021/baseline.json")
    args = parser.parse_args()

    with open(args.results_path, 'r') as f:
        results = json.load(f)

    f1, skip_cnt = f1_score(results)
    print("F1 Score=", f1, "skipping", skip_cnt)
