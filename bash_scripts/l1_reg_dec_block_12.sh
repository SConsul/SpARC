python scripts/baseline.py \
    --train_path "./beliefbank-data-sep2021/qa_train.json" \
    --max_epochs 10 \
    --batch_size 16 \
    --model_path "runs/l1_reg_dec_block_12" \
    --l1_reg 0.00001 \
    --adapter \
    --layer_names "decoder.block.12.layer.2.layer_norm";

# touch runs/l1_reg_dec_block_12/results.txt;

python scripts/inference.py \
    --in_path "./beliefbank-data-sep2021/qa_test.json" \
    --out_path "runs/l1_reg_dec_block_12/l1_reg_dec_block_12_acc.json" \
    --model_path "runs/l1_reg_dec_block_12/10.bin" \
    --adapter \
    --batch_size 512;

python scripts/utils/accuracy.py \
    --results_path "runs/l1_reg_dec_block_12/l1_reg_dec_block_12_acc.json";

python scripts/inference.py \
    --in_path "./beliefbank-data-sep2021/qa_consistency.json" \
    --out_path "runs/l1_reg_dec_block_12/l1_reg_dec_block_12_consis.json" \
    --model_path "runs/l1_reg_dec_block_12/10.bin" \
    --adapter \
    --batch_size 512;

python scripts/utils/consistency_v2.py \
    --results_path "runs/l1_reg_dec_block_12/l1_reg_dec_block_12_consis.json";