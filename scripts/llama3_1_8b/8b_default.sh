
export PYTHONPATH=/home/simarora/code/lolcats/

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 /home/simarora/code/lolcats/llama_recipes/distill_llama.py \
    --model_config llama3_1_8b/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_8b/distill_xent0_mse1000_lr1e-2 \
    --finetune_config llama3_1_8b/finetune_lora_qkvo_alpaca_clean \
    --eval_config eval_alpaca_clean \
    --lk_zero_init \
    --verbose --seed 0 --replicate 0 \
    --eval_steps 100 --dataset_chunk_size 512 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing

# python distill_llama.py \
#     --model_config llama3_1_8b/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01 \
#     --distill_config llama3_1_8b/distill_xent0_mse1000_lr1e-2 \
#     --finetune_config llama3_1_8b/finetune_lora_qkvo_alpaca_clean \
#     --eval_config eval_alpaca_clean \
#     --lk_zero_init \
#     --verbose --seed 0 --replicate 0


# CUDA_VISIBLE_DEVICES=1 python distill_llama.py \
#     --model_config distill_llama3_1_8b_lk_smd_wtk64_fd64_w01 \
#     --distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
#     --finetune_config finetune_lora_qkvo_alpaca_clean \
#     --eval_config eval_alpaca_clean \
#     --lk_zero_init \
#     --verbose --seed 0 --replicate 0
