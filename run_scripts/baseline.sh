python inference.py \
    --in_path "../beliefbank-data-sep2021/qa_test.json" \
    --out_path "runs/baseline.json" \
    --batch_size 16;

python utils/accuracy.py \
    --results_path "runs/baseline.json";

python utils/consistency.py \
    --results_path "runs/baseline.json";