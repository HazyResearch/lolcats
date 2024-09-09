
export PYTHONPATH=/home/simarora/code/lolcats/  # TODO: Change to your path

# Shard the original 70B Llama model into 8 chunks of 10 layers per model below.
torchrun --nnodes 1 --nproc_per_node 8 llama_recipes/trenchcoat_lolcat/save_llama_attn_inputs.py \
    --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_70b/distill_llama_70b_xent0_mse1000_lr1e-2 \
    --finetune_config llama3_1_70b/finetune_llama_70b \
    --verbose --replicate 0 --seed 0 \
    --dataset_chunk_size 1024 \
    --layers_per_model 10 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing \


