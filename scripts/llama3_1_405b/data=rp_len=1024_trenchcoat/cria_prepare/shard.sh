
########################################################################
# This script splits Llama 70B's 80 Transformer layers into individual .pt files of "layers_per_model" layers each.
########################################################################


export PYTHONPATH=/home/simarora/code/lolcats/  #  TODO change to your folder

# Save the model shards: 14 chunks of 9 layers
CUDA_VISIBLE_DEVICES=1 python llama_recipes/trenchcoat_lolcat/shard_llama_to_cria.py \
    --model_config llama3_1_405b/distill_llama3_1_405b_lk_smd_wtk64_fd64_w01 \
    --distill_config llama3_1_405b/rp_distill_llama_405b_xent1_mse1000_lr1e-2 \
    --finetune_config llama3_1_405b/rp_finetune_llama_40b_qv_hparams \
    --verbose --replicate 0 --seed 0 \
    --layer_idx 0 --layers_per_model 9 --device 0

