

export PYTHONPATH=/home/simarora/code/lolcats/       # TODO change to your folder

# Stage 1: Attention distillation 
torchrun --nnodes 1 --nproc_per_node 8 llama_recipes/distill_llama.py \
    --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_70b/distill_rp_llama_70b_xent0_mse1000_lr1e-2 \
    --finetune_config llama3_1_70b/finetune_rp_llama_70b_qkvo \
    --eval_config eval_alpaca_clean \
    --verbose --replicate 0 --seed 0 \
    --lk_zero_init \
    --eval_steps 50 --dataset_chunk_size 1024 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing


# Stage 2: LoRA end-to-end finetuning
torchrun --nnodes 1 --nproc_per_node 8 /home/simarora/code/lolcats/llama_recipes/distill_llama_finetune.py \
    --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_70b/distill_rp_llama_70b_xent0_mse1000_lr1e-2 \
    --finetune_config llama3_1_70b/finetune_rp_llama_70b_qkvo \
    --eval_config eval_alpaca_clean \
    --verbose --replicate 0 --seed 0 \
    --lk_zero_init \
    --eval_steps 50 --dataset_chunk_size 1024 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing

###################
# Sample of how to combine all the sharded model files (__0_0.distcp  __1_0.distcp  __2_0.distcp  __3_0.distcp  __4_0.distcp  __5_0.distcp  __6_0.distcp  __7_0.distcp) into a single .pt file
# NOTE: Please update the checkpoint directories and configs for your specific experiment. The code will printout the checkpoint_paths as well at the end of training:
####################
# torchrun --nnodes=1 \
#     --nproc_per_node=8 \
#     llama_recipes/combine_shards_to_pt.py \
#     --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
#     --distill_config llama3_1_70b/distill_rp_llama_70b_xent0_mse1000_lr1e-2 \
#     --final_finetune_config llama3_1_70b/finetune_rp_llama_70b_qkvo \
#     --verbose --replicate 0 --seed 0 \
#     --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing \
#     --attn_mlp_checkpoint_path /home/simarora/code/lolcats/checkpoints/llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01/distill-dl-d=llama3_1_70b/distill_rp_llama_70b_xent0_mse1000_lr1e-2-m=llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01-f=llama3_1_70b/finetune_rp_llama_70b-fac=1-dcs=1024-se=0-re=0-lzi=1 \
#     --finetune_checkpoint_path /home/simarora/code/lolcats/checkpoints/llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01/finetune-dl-d=llama3_1_70b/distill_rp_llama_70b_xent0_mse1000_lr1e-2-m=llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01-f=llama3_1_70b/finetune_rp_llama_70b_qkvo-fac=1-dcs=1024-se=0-re=0-lzi=1-dcs=1024-se=0-re=0


# Use the ckpt: You can then take your .pt file and run the 70b demo script in demos/llm_mmlu_eval/ etc!


