
# running on 0926

export PYTHONPATH=/home/rahul/code/clean/lolcats/  # Change to your path.

# Save the model shards.
CUDA_VISIBLE_DEVICES=6 python llama_recipes/trenchcoat_lolcat//distill_mini_train.py \
--model_config llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01 \
--distill_config llama3_1_405b/distill_rpcontig2048_dcs2048_xent0_mse1000_lr1e-2 \
--finetune_config llama3_1_405b/rp_contig_finetune_llama_405b_qv_hparams_2048 \
--verbose --replicate 0 --seed 0 \
--layer_idx 54 --layers_per_model 3 --device 0  

