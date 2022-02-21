import json
import argparse

def F1_score(results):
    res = json.load(open(results))
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    skip_count = 0
    for pred in res:
        if pred['pred'].split()[2][:3]=="yes":
            if pred['tgt'].split()[2]=='yes':
                TP+=1
            else:
                FP+=1
        elif pred['pred'].split()[2][:2]=="no":
            if pred['tgt'].split()[2]=='yes':
                FN+=1
            else:
                TN+=1
        else:
            skip_count+=1
    return TP/(TP+0.5*(FP+FN)), skip_count

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', default="./beliefbank-data-sep2021/baseline.json")
    args = parser.parse_args()
    f1, skip_cnt = F1_score(args.results_path)
    print("F1 Score=",f1,"skipping",skip_cnt)