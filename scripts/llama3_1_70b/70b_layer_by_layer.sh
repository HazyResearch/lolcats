

export PYTHONPATH=/home/simarora/code/lolcats/

# Save the layer-by-layer outputs
torchrun --nnodes 1 --nproc_per_node 8 /home/simarora/code/lolcats/llama_recipes/save_outputs.py \
    --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_70b/distill_llama_70b_xent1_mse1000_lr1e-2 \
    --finetune_config llama3_1_70b/finetune_llama_70b \
    --eval_config eval_alpaca_clean \
    --lk_zero_init \
    --verbose --seed 0 --replicate 0 \
    --eval_steps 100 --dataset_chunk_size 512 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing


