
export PYTHONPATH=/home/simarora/code/lolcats/

# Stage 1: Attention distillation 
CUDA_VISIBLE_DEVICES=5 python distill_llama.py \
    --model_config llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd256_w01_attn4layers \
    --distill_config llama3_1_8b/distill_c4_clean_xent1_mse1000_lr1e-2 \
    --finetune_config llama3_1_8b/finetune_qkvo_c4_clean \
    --eval_config eval_alpaca_clean \
    --lk_zero_init --verbose --seed 0 --replicate 800 --state_chunk_len 1024 \
    --num_train_epochs 2 

