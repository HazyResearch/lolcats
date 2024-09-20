

export PYTHONPATH=/home/simarora/code/lolcats/

# evaluate with my pipeline

# # xent-1 mse-1000 red pajama
CUDA_VISIBLE_DEVICES=6,7 torchrun --nnodes 1 --nproc_per_node 2 llama_recipes/evals/eval_mmlu.py \
    --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_70b/distill_rp_llama_70b_xent1_mse1000_lr1e-2 \
    --finetune_config llama3_1_70b/finetune_rp_llama_70b \
    --eval_config eval_alpaca_clean \
    --verbose --replicate 0 --seed 0 \
    --lk_zero_init \
    --eval_steps 100 --dataset_chunk_size 1024 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing


