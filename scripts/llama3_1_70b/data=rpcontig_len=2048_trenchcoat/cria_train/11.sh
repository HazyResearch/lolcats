
export PYTHONPATH=/home/rahul/code/clean/lolcats/  # Change to your path.

# Save the model shards.
CUDA_VISIBLE_DEVICES=1 python llama_recipes/trenchcoat_lolcat//distill_mini_train.py \
    --model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_70b/distill_rp2048_llama_70b_xent0_mse1000_lr1e-2 \
    --finetune_config llama3_1_70b/finetune_rp2048_llama70b_qv \
    --verbose --replicate 0 --seed 0 \
    --layer_idx 50 --layers_per_model 5 --device 0 


