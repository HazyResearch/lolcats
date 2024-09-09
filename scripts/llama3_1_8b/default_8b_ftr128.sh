

export PYTHONPATH=/home/simarora/code/lolcats/

CUDA_VISIBLE_DEVICES=1 python distill_llama.py \
    --model_config ablations/distill_llama3_1_8b_ftr_dim_128 \
    --distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
    --finetune_config finetune_lora_qkvo_alpaca_clean \
    --eval_config eval_alpaca_clean \
    --lk_zero_init \
    --verbose --seed 0 --replicate 0

