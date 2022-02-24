python baseline.py \
    --train_path "../beliefbank-data-sep2021/qa_train.json" \
    --max_epochs 10 \
    --batch_size 4 \
    --model_path "runs/finetune";

python inference.py \
    --in_path "../beliefbank-data-sep2021/qa_test.json" \
    --out_path "runs/finetune/finetune.json" \
    --model_path "runs/finetune/9.bin" \
    --batch_size 16;

python utils/accuracy.py \
    --results_path "runs/finetune/finetune.json";

python utils/consistency.py \
    --results_path "runs/finetune/finetune.json";