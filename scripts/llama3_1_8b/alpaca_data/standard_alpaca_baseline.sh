

export PYTHONPATH=/home/simarora/code/lolcats/  # TODO: Change this to your lolcats path

CUDA_VISIBLE_DEVICES=0 python distill_llama.py \
    --model_config llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_8b/distill_alpaca_clean_xent1_mse1000_lr1e-2 \
    --finetune_config llama3_1_8b/finetune_qkvo_alpaca_clean \
    --eval_config eval_rp_clean \
    --lk_zero_init --verbose --seed 0 --replicate 800 --state_chunk_len 1024 \
    --num_train_epochs 2 

#########
# NOTE
# If you add an argument like follows, it will just start the finetuning stage from the distilled checkpoint, 
# otherwise it will run both stages.
# 
#     --load_distill_checkpoint /home/simarora/code/lolcats/checkpoints/llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01/dl-d=llama3_1_8b/distill_rp_clean_xent1_mse1000_lr1e-2-m=llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01-f=llama3_1_8b/finetune_qkvo_rp_clean-s=0-nte=2-se=0-re=800-scl=1024-lzi=1_distill.pt \
#########

