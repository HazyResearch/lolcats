

export PYTHONPATH=/home/simarora/code/lolcats/

# Stage 1: Attention distillation 
CUDA_VISIBLE_DEVICES=5 python distill_llama.py \
    --model_config llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01_attn4layers \
    --distill_config llama3_1_8b/distill_alpaca_clean_xent1_mse1000_lr1e-2 \
    --finetune_config llama3_1_8b/finetune_qkvo_alpaca_clean \
    --load_distill_checkpoint /home/simarora/code/lolcats/checkpoints/llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01_attn4layers/dl-d=llama3_1_8b/distill_alpaca_clean_xent1_mse1000_lr1e-2-m=llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01_attn4layers-f=llama3_1_8b/finetune_qkvo_alpaca_clean-s=0-nte=2-se=0-re=800-scl=1024-lzi=1_distill.pt \
    --eval_config eval_alpaca_clean \
    --lk_zero_init --verbose --seed 0 --replicate 800 --state_chunk_len 1024 \
    --num_train_epochs 2 
