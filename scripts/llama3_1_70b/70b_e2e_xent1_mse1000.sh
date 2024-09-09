

export PYTHONPATH=/home/simarora/code/lolcats/

# Stage 1: Attention distillation 
torchrun --nnodes 1 --nproc_per_node 8 /home/simarora/code/lolcats/llama_recipes/distill_llama.py \
    --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_70b/distill_llama_70b_xent1_mse1000_lr1e-2 \
    --finetune_config llama3_1_70b/finetune_llama_70b \
    --eval_config eval_alpaca_clean \
    --verbose --replicate 0 --seed 0 \
    --lk_zero_init \
    --eval_steps 16 --dataset_chunk_size 1024 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing

# Stage 2: LoRA end-to-end finetuning
# with qk lora
torchrun --nnodes 1 --nproc_per_node 8 /home/simarora/code/lolcats/llama_recipes/distill_llama_finetune.py \
    --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_70b/distill_llama_70b_xent1_mse1000_lr1e-2 \
    --finetune_config llama3_1_70b/finetune_llama_70b \
    --eval_config eval_alpaca_clean \
    --verbose --replicate 0 --seed 0 \
    --lk_zero_init \
    --eval_steps 100 --dataset_chunk_size 1024 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing


# As an example, if you would like to later change how you finetune, you can hardcode the checkpoint path; by default our repo will look for checkpoints at a path that's determined by the config names initially used to train. So if you change the configs to new names, you should use this approach:
# Stage 2 alternate: Finetune with qkvo lora
# torchrun --nnodes 1 --nproc_per_node 8 /home/simarora/code/lolcats/llama_recipes/distill_llama_finetune.py \
#     --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
#     --distill_config llama3_1_70b/distill_llama_70b_xent1_mse1000_lr1e-2 \
#     --finetune_config llama3_1_70b/finetune_llama_70b_qkvo \
#     --eval_config eval_alpaca_clean \
#     --verbose --replicate 0 --seed 0 \
#     --lk_zero_init \
#     --load_distill_checkpoint /home/simarora/code/lolcats/checkpoints/llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01/distill-dl-d=llama3_1_70b/distill_llama_70b_xent1_mse1000_lr1e-2-m=llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01-f=llama3_1_70b/finetune_llama_70b-fac=1-dcs=1024-se=0-re=0-lzi=1 \
#     --eval_steps 100 --dataset_chunk_size 1024 \
#     --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing


# Stage 3: Eval the end perplexity on the alpaca eval set again if you'd like (it should also log during your above training runs)
torchrun --nnodes 1 --nproc_per_node 8 llama_recipes/evals/eval_ppl.py \
    --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_70b/distill_llama_70b_xent1_mse1000_lr1e-2 \
    --finetune_config llama3_1_70b/finetune_llama_70b \
    --eval_config eval_alpaca_clean \
    --verbose --replicate 0 --seed 0 \
    --lk_zero_init \
    --eval_steps 100 --dataset_chunk_size 1024 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing

    