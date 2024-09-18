
# # 128 dim attention model
# CUDA_VISIBLE_DEVICES=2 python lm_eval_harness/eval_lm_harness.py \
# --model_type lolcats_ckpt \
# --attn_mlp_checkpoint_path /home/simarora/code/lolcats//checkpoints/llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd128_w01_attn4layers/dl-d=llama3_1_8b/distill_c4_clean_xent1_mse1000_lr1e-2-m=llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd128_w01_attn4layers-f=llama3_1_8b/finetune_qkvo_c4_clean-s=0-nte=2-se=0-re=800-scl=1024-lzi=1_distill.pt \
# --finetune_checkpoint_path /home/simarora/code/lolcats//checkpoints/llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd128_w01_attn4layers/dl-d=llama3_1_8b/distill_c4_clean_xent1_mse1000_lr1e-2-m=llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd128_w01_attn4layers-f=llama3_1_8b/finetune_qkvo_c4_clean-s=0-nte=2-se=0-re=800-scl=1024-lzi=1-bs=1-gas=8-nte=2-ms=2500-se=0-re=800_ft.pt \
# --task mmlu --num_shots 5 --no_cache --verbose --limit 5


# # 256 dim attention model
# CUDA_VISIBLE_DEVICES=3 python lm_eval_harness/eval_lm_harness.py \
# --model_type lolcats_ckpt \
# --attn_mlp_checkpoint_path /home/simarora/code/lolcats/checkpoints/llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd256_w01_attn4layers/dl-d=llama3_1_8b/distill_c4_clean_xent1_mse1000_lr1e-2-m=llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd256_w01_attn4layers-f=llama3_1_8b/finetune_qkvo_c4_clean-s=0-nte=2-se=0-re=800-scl=1024-lzi=1_distill.pt \
# --finetune_checkpoint_path /home/simarora/code/lolcats/checkpoints/llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd256_w01_attn4layers/dl-d=llama3_1_8b/distill_c4_clean_xent1_mse1000_lr1e-2-m=llama3_1_8b/distill_llama_3_8b_lk_smd_wtk64_fd256_w01_attn4layers-f=llama3_1_8b/finetune_qkvo_c4_clean-s=0-nte=2-se=0-re=800-scl=1024-lzi=1-bs=1-gas=8-nte=2-ms=2500-se=0-re=800_ft.pt \
# --task mmlu --num_shots 5 --no_cache --verbose --limit 5


