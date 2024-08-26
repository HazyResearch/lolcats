
export PYTHONPATH=/home/simarora/code/lolcats/

torchrun --nnodes 1 --nproc_per_node 8 /home/simarora/code/lolcats/llama_recipes/distill_llama.py \
    --model_config distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
    --distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
    --finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_70b \
    --eval_config eval_alpaca_clean \
    --verbose --replicate 0 --seed 0 \
    --lk_zero_init \
    --eval_steps 100 --dataset_chunk_size 512 \
    --enable_fsdp --low_cpu_fsdp 

# torchrun --nnodes 1 --nproc_per_node 8 llama_recipes/distill_llama_finetune.py \
#     --model_config distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
#     --distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
#     --finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_70b \
#     --eval_config eval_alpaca_clean \
#     --verbose --replicate 0 --seed 0 \
#     --lk_zero_init \
#     --eval_steps 100 --dataset_chunk_size 512 \
#     --enable_fsdp --low_cpu_fsdp
    