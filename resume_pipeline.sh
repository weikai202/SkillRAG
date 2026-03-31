#!/bin/bash
# Resume: train layer 31, then re-run probing + skillrag eval

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
export WANDB_MODE=disabled

echo "============================================================"
echo "Train layer 31 (all 3 datasets, last=trivia wins checkpoint)"
echo "============================================================"
for ds in nq hotpotqa trivia; do
    echo "  Training $ds layer 31 ..."
    CUDA_VISIBLE_DEVICES=0 python train.py --method tokens_mean --batch_size 6 --lr 0.001 --layer 31 --device cuda:0 --epochs 2 --model_id "$MODEL" --dataset_name "$ds" --train_ds_ratio 1.0
    echo "  $ds done."
done

echo ""
echo "============================================================"
echo "Re-run probing + skillrag eval (none + simple already done)"
echo "============================================================"
EVAL="--steps_limit 200 --tr_or_dev dev --is_cot --is_sparse --model_id $MODEL --ds 3 --position resid_post --threshold 0.0 --max_retrieval_rounds 3 --extracting_cot_qa --extract_sep --sep_number 0"

for ds in nq musique hotpotqa trivia 2wikimultihopqa; do
    CUDA_VISIBLE_DEVICES=0 python exp_rag.py --retr_method probing --dataset_name "$ds" $EVAL &
    CUDA_VISIBLE_DEVICES=1 python exp_rag.py --retr_method skillrag --dataset_name "$ds" $EVAL &
    wait
done

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
