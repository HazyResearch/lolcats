
# take the finetuned checkpoints and save into a single .pt file

export PYTHONPATH=/home/rahul/code/clean/lolcats/  # Change to your path.

torchrun --nnodes 1 --nproc_per_node 8 llama_recipes/save_fsdp_to_pt_405b.py \
--model_config llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01 \
--distill_config llama3_1_405b/distill_rpcontig2048_dcs2048_xent0_mse1000_lr1e-2 \
--finetune_config llama3_1_405b/rp_contig_finetune_llama_405b_qv_hparams_2048 \
--final_finetune_config llama3_1_405b/finetune_llama_405b_qkvo_e2_rp_2048 \
--verbose --replicate 0 --seed 0 \
--layers_per_model 3 --layer_idx 0 \
--enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing \
--load_finetune_checkpoint /home/rahul/code/clean/lolcats/checkpoints/llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01/sharded_layers/finetune-dl-d=llama3_1_405b/distill_rpcontig2048_dcs2048_xent0_mse1000_lr1e-2-m=llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01-f=llama3_1_405b/rp_contig_finetune_llama_405b_qv_hparams_2048-s=0-se=0-re=0-se=0-re=0_v0

