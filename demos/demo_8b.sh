
CONFIG_DIR='/home/bfs/simran/attention/lolcats/configs/'   # update to your path

# Using huggingface checkpoints 
CUDA_VISIBLE_DEVICES=0 python -Wignore demo_lolcats_hf.py \
    --model_config_path ${CONFIG_DIR}/model/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01.yaml \
    --distill_config_path ${CONFIG_DIR}/experiment/distill_alpaca_clean_xent0_mse1000_lr1e-2.yaml \
    --finetune_config_path ${CONFIG_DIR}/experiment/finetune_lora_qkvo_alpaca_clean.yaml \
    --attn_mlp_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-distill' \
    --finetune_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-ft-lora' \
    --num_generations 1 \
    --max_new_tokens 50


# Reference script:
# if you train your own LoLCATS weights, you can use the following command to run inference with your local checkpoints:
# CHECKPOINT_DIR='/home/mzhang/projects/lolcats/checkpoints/'
# CUDA_VISIBLE_DEVICES=0 python -Wignore demo_lolcats_hf.py \
#     --model_config_path ${CONFIG_DIR}/model/llama3_1_8b/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01.yaml \
#     --distill_config_path ${CONFIG_DIR}/experiment/llama3_1_8b/distill_alpaca_clean_xent0_mse1000_lr1e-2.yaml \
#     --finetune_config_path ${CONFIG_DIR}/experiment/llama3_1_8b/finetune_qkvo_alpaca_clean.yaml \
#     --attn_mlp_checkpoint_path ${CHECKPOINT_DIR}/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2-m=distill_llama3_1_8b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean-s=0-ms=2500-se=0-re=100-lzi=1_distill.pt \
#     --finetune_checkpoint_path ${CHECKPOINT_DIR}/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2-m=distill_llama3_1_8b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean-s=0-ms=2500-se=0-re=100-lzi=1-bs=1-gas=8-nte=2-ms=2500-se=0-re=100_ft.pt \
#     --num_generations 1



