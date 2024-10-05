
export PYTHONPATH=/home/rahul/code/clean/lolcats/   #  TODO change to your folder

CUDA_VISIBLE_DEVICES=1 python llama_recipes/trenchcoat_lolcat/shard_llama_to_cria.py \
--model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
--distill_config llama3_1_70b/distill_rpcontig2048_dcs2048_xent0_mse1000_lr1e-2 \
--finetune_config llama3_1_70b/rp_contig_finetune_llama_70b_qv_hparams_2048 \
--verbose --replicate 0 --seed 0 \
--layer_idx 0 --layers_per_model 5 --device 2 --fsdp_cpu_offload 





