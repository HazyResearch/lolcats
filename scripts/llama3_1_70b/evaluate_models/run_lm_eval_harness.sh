

CHECKPINT_DIR='/home/simarora/code/lolcats/checkpoints/' # TODO: update to your path

export PYTHONPATH=/home/simarora/code/lolcats/ # TODO: update to your path

CUDA_VISIBLE_DEVICES=0,1,2,3 python lm_eval_harness/eval_lm_harness_big.py \
--model_config llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
--distill_config llama3_1_70b/distill_rp_llama_70b_xent1_mse1000_lr1e-2  \
--finetune_config llama3_1_70b/finetune_rp_llama_70b \
--enable_fsdp --low_cpu_fsdp \
--verbose --replicate 0 --seed 0 --lk_zero_init \
--task hendrycksTest --num_shots 5 --no_cache \
--attn_mlp_checkpoint_path ${CHECKPINT_DIR}/llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01/distill-dl-d=llama3_1_70b/distill_rp_llama_70b_xent1_mse1000_lr1e-2-m=llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01-f=llama3_1_70b/finetune_rp_llama_70b-fac=1-dcs=1024-se=0-re=0-lzi=1 \  # TODO: update to your checkpoint path
--finetune_checkpoint_path ${CHECKPINT_DIR}/llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01/finetune-dl-d=llama3_1_70b/distill_rp_llama_70b_xent1_mse1000_lr1e-2-m=llama3_1_70b/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01-f=llama3_1_70b/finetune_rp_llama_70b_qkvo-fac=1-dcs=1024-se=0-re=0-lzi=1-dcs=1024-se=0-re=0  # TODO: update to your checkpoint path


