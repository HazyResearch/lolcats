# Llama Recipes to Linearize 70B and 405B LLMs

This directory contains code modified from the great [llama-recipes](https://github.com/meta-llama/llama-recipes/tree/main/src/llama_recipes) repository that we use to linearize the 70B and 405B LLMs.

- For more info on supporting files, please see the original docs at
  https://github.com/meta-llama/llama-recipes/tree/main/docs
- Our additional files (the relevant ones) are:
  - `distill_llama.py`
  - `distill_llama_finetune.py`
  - `trainer_attention.py`
  - `trainer_finetune.py`

More details and sample commands below, but the code borrows the Fully Sharded Data Parallel (FSDP) pipeline from llama-recipes to linearizing Llama 70B and 405B models in bfloat16 precision and [multiple GPUs](https://github.com/meta-llama/llama-recipes/blob/main/docs/multi_gpu.md). We separate the two stages of LoLCATs linearizing into two scripts:

1. `distill_llama.py`: where we first train subquadratic attentions to mimic the softmax attentions (saving the learned attention feature map checkpoints)
2. `distill_llama_finetune.py`: where we swap in the learned attentions and finetune the rest of the model with LoRA (saving the LoRA checkpoints)

By passing in the same configurations files and arguments to both scripts, `distill_llama_finetune.py` should automatically load the saved checkpoints and pick up from where `distill_llama.py` left off.

### Sample Commands

**_Script 1: Attention Transfer_**

```bash
torchrun --nnodes 1 --nproc_per_node 4 \
llama_recipes/distill_llama.py \
--model_config distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_70b \
--eval_config eval_alpaca_clean \
--verbose --replicate 0 --seed 0 \
--enable_fsdp --low_cpu_fsdp
```

**_Script 2: Low-rank Adaptation_**

```bash
torchrun --nnodes 1 --nproc_per_node 4 \
llama_recipes/distill_llama_finetune.py \
--model_config distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_70b \
--eval_config eval_alpaca_clean \
--verbose --replicate 0 --seed 0 \
--enable_fsdp --low_cpu_fsdp
```

### Training Requirements

See https://huggingface.co/blog/llama31#training-memory-requirements