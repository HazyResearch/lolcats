
export PYTHONPATH=/home/rahul/code/clean/lolcats/         # TODO change to your folder

# Stitch the blocks together into a single .pt file
torchrun --nnodes=1 \
    --nproc_per_node=8 \
    llama_recipes/trenchcoat_lolcat/save_fsdp_to_pt.py \
    --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_70b/distill_rp2048_llama_70b_xent0_mse1000_lr1e-2 \
    --finetune_config llama3_1_70b/finetune_rp2048_llama70b_qv \
    --final_finetune_config llama3_1_70b/finetune_rp2048_llama_70b_qkvo \  # will use this config
    --verbose --replicate 0 --seed 0 \
    --layers_per_model 5 --layer_idx 0 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing \
    --load_finetune_checkpoint /path/to/your/block-wise/checkpoints/

# Sample load_finetune_checkpoint:  
# /home/rahul/code/clean/lolcats/checkpoints/llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01/sharded_layers/finetune-dl-d=llama3_1_70b/distill_rpcontig2048_dcs2048_xent0_mse1000_lr1e-2-m=llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01-f=llama3_1_70b/rp_contig_finetune_llama_70b_qv_hparams_2048-s=0-se=0-re=0-se=0-re=0 


# Finetune with LoRA
torchrun --nnodes=1 \
    --nproc_per_node=8 \
    llama_recipes/stitch_mini_finetune.py \
    --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_70b/distill_rp2048_llama_70b_xent0_mse1000_lr1e-2 \
    --finetune_config llama3_1_70b/finetune_rp2048_llama70b_qv \
    --final_finetune_config llama3_1_70b/finetune_rp2048_llama_70b_qkvo \   # will use this config
    --verbose --replicate 0 --seed 0 \
    --layers_per_model 5 --layer_idx 0 \
    --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing




