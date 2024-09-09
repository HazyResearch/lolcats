
export PYTHONPATH=/home/simarora/code//lolcats/

# Save the model shards: 14 chunks of 9 layers
CUDA_VISIBLE_DEVICES=7 python llama_recipes/distill_mini_train.py \
--model_config llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01 \
--distill_config llama3_1_405b/distill_llama_405b_xent1_mse1000_lr1e-2 \
--finetune_config llama3_1_405b/finetune_layer_mini_xent1_mse1000 \
--verbose --replicate 0 --seed 0 \
--load_distill_checkpoint "/home/simarora/code/lolcats/checkpoints/dl-d=llama3_1_405b/distill_llama_405b_xent1_mse1000_lr1e-2-m=llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01-f=llama3_1_405b/finetune_layer_mini_xent1_mse1000-s=0-se=0-re=0-in=063-out=071_distill.pt" \
--layer_idx 63 --layers_per_model 9 --device 0 