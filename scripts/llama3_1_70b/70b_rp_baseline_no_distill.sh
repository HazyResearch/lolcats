# """Runs lora stage on a pretrained model without attention conversion """

export PYTHONPATH=/home/simarora/code/lolcats/

torchrun --nnodes 1 --nproc_per_node 8 llama_recipes/distill_llama_finetune.py \
    --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_70b/no_distill_llama3_1_70b_rp \
    --finetune_config llama3_1_70b/finetune_rp_llama_70b \
    --eval_config eval_alpaca_clean \
    --verbose --replicate 0 --seed 0 \
    --lk_zero_init \
    --eval_steps 100 --dataset_chunk_size 1024 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing


