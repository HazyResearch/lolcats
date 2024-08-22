### Various development scripts

For your viewing pleasure.

**_Initial testing of the distillation recipe_**

May need to prepend with `NCCL_CUMEM_ENABLE=0`

```bash
torchrun --nnodes 1 --nproc_per_node 7 \
llama_recipes/distill_llama.py \
--model_config distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_70b \
--eval_config eval_alpaca_clean \
--verbose --replicate 0 --seed 0 \
--enable_fsdp --low_cpu_fsdp \
--dataset_chunk_size 1024 \
--attention_type lolcats_llama_window_tk_bf16 \
--eval_steps 1 --dataset_chunk_size 256
```

```bash
torchrun --nnodes 1 --nproc_per_node 7 \
llama_recipes/distill_llama_finetune.py \
--model_config distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_70b \
--eval_config eval_alpaca_clean \
--verbose --replicate 0 --seed 0 \
--enable_fsdp --low_cpu_fsdp \
--dataset_chunk_size 1024 \
--attention_type lolcats_llama_window_tk_bf16 \
--eval_steps 1 --dataset_chunk_size 256
```

** fd32 **

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nnodes 1 --nproc_per_node 8 \
llama_recipes/distill_llama.py \
--model_config distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_llama3_1_70b_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_70b \
--eval_config eval_alpaca_clean \
--verbose --replicate 0 --seed 0 \
--enable_fsdp --low_cpu_fsdp \
--dataset_chunk_size 512 \
--attention_type lolcats_llama_window_tk_bf16 \
--eval_steps 1
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nnodes 1 --nproc_per_node 8 \
llama_recipes/distill_llama_finetune.py \
--model_config distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_llama3_1_70b_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_70b \
--eval_config eval_alpaca_clean \
--verbose --replicate 0 --seed 0 \
--enable_fsdp --low_cpu_fsdp \
--dataset_chunk_size 512 \
--attention_type lolcats_llama_window_tk_bf16
```

** fd32 **

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nnodes 1 --nproc_per_node 7 \
llama_recipes/distill_llama.py \
--model_config distill_llama3_1_70b_lk_smd_fd32 \
--distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_wqkvo_alpaca_clean_llama3_1_70b \
--eval_config eval_alpaca_clean \
--verbose --replicate 0 --seed 0 \
--enable_fsdp --low_cpu_fsdp \
--dataset_chunk_size 1024 \
--attention_type lolcats_llama_window_tk_bf16
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nnodes 1 --nproc_per_node 7 \
llama_recipes/distill_llama_finetune.py \
--model_config distill_llama3_1_70b_lk_smd_wtk64_fd32_w01 \
--distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_wqkvo_alpaca_clean_llama3_1_70b \
--eval_config eval_alpaca_clean \
--verbose --replicate 0 --seed 0 \
--enable_fsdp --low_cpu_fsdp \
--dataset_chunk_size 1024 \
--attention_type lolcats_llama_window_tk_bf16
```

** Sliding Window Hybrid Attention **

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nnodes 1 --nproc_per_node 8 \
llama_recipes/distill_llama.py \
--model_config distill_llama3_1_70b_lk_smd_wsws64_fd64 \
--distill_config distill_alpaca_clean_llama3_1_70b_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_70b \
--eval_config eval_alpaca_clean \
--verbose --replicate 0 --seed 0 \
--enable_fsdp --low_cpu_fsdp \
--dataset_chunk_size 768 \
```

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nnodes 1 --nproc_per_node 8 \
llama_recipes/distill_llama_finetune.py \
--model_config distill_llama3_1_70b_lk_smd_wsws64_fd64 \
--distill_config distill_alpaca_clean_llama3_1_70b_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_70b \
--eval_config eval_alpaca_clean \
--verbose --replicate 0 --seed 0 \
--enable_fsdp --low_cpu_fsdp \
--dataset_chunk_size 768 \
```
