

# Replace the CHECKPOINT_DIR and the checkpoint names with your own paths and names
# The checkpoint names are outputted at the end of the distillation stage

CHECKPOINT_DIR='/home/simarora/code/lolcats/checkpoints/'

CUDA_VISIBLE_DEVICES=4 python lm_eval_harness/eval_lm_harness.py --model_type lolcats_ckpt \
--attn_mlp_checkpoint_path ${CHECKPOINT_DIR}/llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01/dl-d=llama3_1_8b/distill_alpaca_clean_xent1_mse1000_lr1e-2-m=llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01-f=llama3_1_8b/finetune_qkvo_alpaca_clean-s=0-nte=2-se=0-re=800-scl=1024-lzi=1_distill.pt \
--finetune_checkpoint_path ${CHECKPOINT_DIR}/llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01/dl-d=02081_lzi=1_distill1d-m=llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01-f=llama3_1_8b/finetune_qkvo_alpaca_clean-s=0-nte=2-se=0-re=800-scl=1024-lzi=1-nte=2-se=0-re=800_ft.pt \
--task mmlu --num_shots 5 --no_cache --verbose


