#!/bin/bash
set -e

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
export WANDB_MODE=disabled

echo "============================================================"
echo "PHASE 3: Training — 2 layers at a time across 2 GPUs"
echo "============================================================"
for ds in nq hotpotqa trivia; do
    echo "  Training $ds ..."
    CUDA_VISIBLE_DEVICES=0 python train.py --method tokens_mean --batch_size 6 --lr 0.001 --layer 12 --device cuda:0 --epochs 2 --model_id "$MODEL" --dataset_name "$ds" --train_ds_ratio 1.0 &
    CUDA_VISIBLE_DEVICES=1 python train.py --method tokens_mean --batch_size 6 --lr 0.001 --layer 16 --device cuda:0 --epochs 2 --model_id "$MODEL" --dataset_name "$ds" --train_ds_ratio 1.0 &
    wait
    CUDA_VISIBLE_DEVICES=0 python train.py --method tokens_mean --batch_size 6 --lr 0.001 --layer 20 --device cuda:0 --epochs 2 --model_id "$MODEL" --dataset_name "$ds" --train_ds_ratio 1.0 &
    CUDA_VISIBLE_DEVICES=1 python train.py --method tokens_mean --batch_size 6 --lr 0.001 --layer 24 --device cuda:0 --epochs 2 --model_id "$MODEL" --dataset_name "$ds" --train_ds_ratio 1.0 &
    wait
    CUDA_VISIBLE_DEVICES=0 python train.py --method tokens_mean --batch_size 6 --lr 0.001 --layer 28 --device cuda:0 --epochs 2 --model_id "$MODEL" --dataset_name "$ds" --train_ds_ratio 1.0 &
    CUDA_VISIBLE_DEVICES=1 python train.py --method tokens_mean --batch_size 6 --lr 0.001 --layer 32 --device cuda:0 --epochs 2 --model_id "$MODEL" --dataset_name "$ds" --train_ds_ratio 1.0 &
    wait
    echo "  $ds done."
done

echo ""
echo "============================================================"
echo "PHASE 4: Evaluation — 2 jobs at a time across 2 GPUs"
echo "============================================================"
EVAL="--steps_limit 200 --tr_or_dev dev --is_cot --is_sparse --model_id $MODEL --ds 3 --position resid_post --threshold 0.0 --max_retrieval_rounds 3 --extracting_cot_qa --extract_sep --sep_number 0"

for ds in nq musique hotpotqa trivia 2wikimultihopqa; do
    CUDA_VISIBLE_DEVICES=0 python exp_rag.py --retr_method none --dataset_name "$ds" $EVAL &
    CUDA_VISIBLE_DEVICES=1 python exp_rag.py --retr_method simple --dataset_name "$ds" $EVAL &
    wait
    CUDA_VISIBLE_DEVICES=0 python exp_rag.py --retr_method probing --dataset_name "$ds" $EVAL &
    CUDA_VISIBLE_DEVICES=1 python exp_rag.py --retr_method skillrag --dataset_name "$ds" $EVAL &
    wait
done

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
