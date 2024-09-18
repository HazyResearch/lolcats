


CUDA_VISIBLE_DEVICES=0 python -Wignore demo_lolcats_llm.py \
--attn_mlp_checkpoint_path '/home/mzhang/projects/lolcats/checkpoints/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2-m=distill_llama3_1_8b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean-s=0-ms=2500-se=0-re=100-lzi=1_distill.pt' \
--finetune_checkpoint_path '/home/mzhang/projects/lolcats/checkpoints/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2-m=distill_llama3_1_8b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean-s=0-ms=2500-se=0-re=100-lzi=1-bs=1-gas=8-nte=2-ms=2500-se=0-re=100_ft.pt' \
--num_generations 1


