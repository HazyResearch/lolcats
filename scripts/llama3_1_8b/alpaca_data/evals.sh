

# standard model
CUDA_VISIBLE_DEVICES=1 python lm_eval_harness/eval_lm_harness.py --model_type lolcats_ckpt \
--attn_mlp_checkpoint_path /home/simarora/code/lolcats/checkpoints/llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01/dl-d=llama3_1_8b/distill_alpaca_clean_xent1_mse1000_lr1e-2-m=llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01-f=llama3_1_8b/finetune_qkvo_alpaca_clean-s=0-nte=2-se=0-re=800-scl=1024-lzi=1_distill.pt \
--finetune_checkpoint_path /home/simarora/code/lolcats/checkpoints/llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01/dl-d=02081_lzi=1_distill1d-m=llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01-f=llama3_1_8b/finetune_qkvo_alpaca_clean-s=0-nte=2-se=0-re=800-scl=1024-lzi=1-nte=2-se=0-re=800_ft.pt \
--task mmlu --num_shots 5 --no_cache --verbose --limit 5


# attention model
# CUDA_VISIBLE_DEVICES=2 python lm_eval_harness/eval_lm_harness.py --model_type lolcats_ckpt \
# --attn_mlp_checkpoint_path /home/simarora/code/lolcats/checkpoints/llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01_attn4layers/dl-d=llama3_1_8b/distill_alpaca_clean_xent1_mse1000_lr1e-2-m=llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01_attn4layers-f=llama3_1_8b/finetune_qkvo_alpaca_clean-s=0-nte=2-se=0-re=800-scl=1024-lzi=1_distill.pt \
# --finetune_checkpoint_path /home/simarora/code/lolcats/checkpoints/llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01_attn4layers/dl-d=02081_lzi=1_distill1d-m=llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd64_w01_attn4layers-f=llama3_1_8b/finetune_qkvo_alpaca_clean-s=0-nte=2-se=0-re=800-scl=1024-lzi=1-nte=2-se=0-re=800_ft.pt \
# --task mmlu --num_shots 5 --no_cache --verbose --limit 5




