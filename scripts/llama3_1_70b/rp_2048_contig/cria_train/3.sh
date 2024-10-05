
export PYTHONPATH=/home/rahul/code/clean/lolcats/  # Change to your path.

# Save the model shards.
CUDA_VISIBLE_DEVICES=3 python llama_recipes/trenchcoat_lolcat//distill_mini_train.py \
--model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
--distill_config llama3_1_70b/distill_rpcontig2048_dcs2048_xent0_mse1000_lr1e-2 \
--finetune_config llama3_1_70b/rp_contig_finetune_llama_70b_qv_hparams_2048 \
--verbose --replicate 0 --seed 0 \
--layer_idx 10 --layers_per_model 5 --device 0 


