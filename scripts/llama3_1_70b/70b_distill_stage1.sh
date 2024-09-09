
export PYTHONPATH=/home/simarora/code/lolcats/

# torchrun --nnodes 1 --nproc_per_node 8 /home/simarora/code/lolcats/llama_recipes/distill_llama.py \
#     --model_config llama3_1_70b/faster_distill_70b \
#     --distill_config llama3_1_70b/distill_llama_70b_xent0_mse1000_lr1e-2 \
#     --finetune_config llama3_1_70b/finetune_llama_70b \
#     --eval_config eval_alpaca_clean \
#     --verbose --replicate 0 --seed 0 \
#     --lk_zero_init \
#     --eval_steps 16 --dataset_chunk_size 1024 \
#     --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing

torchrun --nnodes 1 --nproc_per_node 8 /home/simarora/code/lolcats/llama_recipes/eval_stage_1.py \
    --model_config llama3_1_70b/faster_distill_70b \
    --distill_config llama3_1_70b/distill_llama_70b_xent0_mse1000_lr1e-2 \
    --finetune_config llama3_1_70b/finetune_llama_70b \
    --eval_config eval_alpaca_clean \
    --verbose --replicate 0 --seed 0 \
    --lk_zero_init \
    --eval_steps 16 --dataset_chunk_size 1024 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing

