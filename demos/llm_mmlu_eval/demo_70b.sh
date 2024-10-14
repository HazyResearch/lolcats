export PYTHONPATH=/home/simarora/code/lolcats/

# Use HF checkpoint paths (can also prob get away with 2 GPUs - longer contexts may not fit tho)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes 1 --nproc_per_nodes 8 \
    demos/llm_mmlu_eval/eval_mmlu.py \
    --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_70b/distill_rp_llama_70b_xent0_mse1000_lr1e-2 \
    --finetune_config llama3_1_70b/finetune_rp_llama_70b_qkvo \
    --eval_config eval_alpaca_clean \
    --verbose --replicate 0 --seed 0 \
    --lk_zero_init \
    --eval_steps 100 --dataset_chunk_size 1024 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing \
    --experiment_tag lolcats_hf_70b \
    --finetune_checkpoint_path 'hazyresearch/lolcats-llama-3.1-70b' 

# Example using local paths, in case you train your own model.
# CUDA_VISIBLE_DEVICES=6,7 torchrun --nnodes 1 --nproc_per_node 2 \
#     demos/llm_mmlu_eval/eval_mmlu.py \
#     --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
#     --distill_config llama3_1_70b/distill_rp_llama_70b_xent0_mse1000_lr1e-2 \
#     --finetune_config llama3_1_70b/finetune_rp_llama_70b_qkvo \
#     --eval_config eval_alpaca_clean \
#     --verbose --replicate 0 --seed 0 \
#     --lk_zero_init \
#     --eval_steps 100 --dataset_chunk_size 1024 \
#     --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing \
#     --experiment_tag my_lolcats_70b \
#     --finetune_checkpoint_path ckpt_lora-dl-d=distill_rp_llama_70b_xent0_mse1000_lr1e-2-m=distill_llama3_1_70b_lk_smd_wtk64_fd64_w01-f=finetune_rp_llama_70b_qkvo-fac=1-se=0-re=0-se=0-re=0.pt

