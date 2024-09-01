
export PYTHONPATH=/home/simarora/code//lolcats/

torchrun --nnodes 1 --nproc_per_node 8 \
llama_recipes/save_llama_attn_inputs.py \
--model_config llama3_1_8b/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01 \
--distill_config llama3_1_8b/distill_xent0_mse1000_lr1e-2 \
--finetune_config llama3_1_8b/finetune_lora_qkvo_alpaca_clean \
--layers_per_model 8 \
--verbose --replicate 0 --seed 0 \
--enable_fsdp --low_cpu_fsdp

CUDA_VISIBLE_DEVICES=4 python distill_llama_mini.py \
--model_config llama3_1_8b/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01 \
--distill_config llama3_1_8b/distill_xent0_mse1000_lr1e-2 \
--finetune_config llama3_1_8b/finetune_layer_mini_xent1_mse1000 \
--layer_idx 0 --layers_per_model 8 --device 0 \
--verbose --seed 0 --replicate 0 




# My approach

# Save the layer-by-layer outputs
# torchrun --nnodes 1 --nproc_per_node 8 /home/simarora/code/lolcats/llama_recipes/save_outputs.py \
#     --model_config llama3_1_8b/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01 \
#     --distill_config llama3_1_8b/distill_xent0_mse1000_lr1e-2 \
#     --finetune_config llama3_1_8b/finetune_lora_qkvo_alpaca_clean \
#     --eval_config eval_alpaca_clean \
#     --lk_zero_init \
#     --verbose --seed 0 --replicate 0 \
#     --eval_steps 100 --dataset_chunk_size 512 \
#     --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing

# Launch layer-by-layer training (use 1 node so each layer can be handled by diff nodes)
# torchrun --nnodes 1 --nproc_per_node 1 /home/simarora/code/lolcats/llama_recipes/train_layer_by_layer.py \
#     --model_config llama3_1_8b/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01 \
#     --distill_config llama3_1_8b/distill_xent0_mse1000_lr1e-2 \
#     --finetune_config llama3_1_8b/finetune_lora_qkvo_alpaca_clean \
#     --eval_config eval_alpaca_clean \
#     --lk_zero_init \
#     --verbose --seed 0 --replicate 0 \
#     --eval_steps 100 --dataset_chunk_size 512 \
#     --enable_fsdp --low_cpu_fsdp --fsdp_activation_checkpointing

