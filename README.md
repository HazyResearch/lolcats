# LoLCATS in a Trenchcoat

<p align="center">
<img src="assets/hedgehog_llamas_big.png" align='center' width=35% height=35%>
</p>

This readme describes how to obtain scaled up, 70B and 405B parameter, LoLCATS!

---

### Getting started

Please see the main branch README for: 
- Setup and installation instructions. 
- Details on how the experimental config files are oriented (yaml structure).
- Instructions to install optional CUDA kernels.

---

### Overview: Linearizing large language models.

We support linearizing larger LLMs (Llama 3.1 70B, Llama 3.1 405B) using the great [llama-recipes](https://github.com/meta-llama/llama-recipes/tree/main/src/llama_recipes) repository.

See `llama_recipes/README.md` for more details. At a high-level, we borrow the Fully Sharded Data Parallel (FSDP) pipeline, linearize **unquantized** models, and split the two stages of LoLCATs linearizing into two scripts:

1. `distill_llama.py`: where we first train subquadratic attentions to mimic the softmax attentions (saving the learned attention feature map checkpoints)
2. `distill_llama_finetune.py`: where we swap in the learned attentions and finetune the rest of the model with LoRA (saving the LoRA checkpoints)

By passing in the same configurations files and arguments to both scripts, `distill_llama_finetune.py` should automatically load the saved checkpoints and pick up from where `distill_llama.py` left off.

#### Sample Commands

**_Script 1: Attention Transfer_**

```bash
torchrun --nnodes 1 --nproc_per_node 8 \
llama_recipes/distill_llama.py \
--model_config distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_70b \
--eval_config eval_alpaca_clean \
--lk_zero_init \
--verbose --replicate 0 --seed 0 \
--enable_fsdp --low_cpu_fsdp
```

**_Script 2: Low-rank Adaptation_**

```bash
torchrun --nnodes 1 --nproc_per_node 8 \
llama_recipes/distill_llama_finetune.py \
--model_config distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_70b \
--eval_config eval_alpaca_clean \
--lk_zero_init \
--verbose --replicate 0 --seed 0 \
--enable_fsdp --low_cpu_fsdp
```

#### GPU Memory Training Requirements

See https://huggingface.co/blog/llama31#training-memory-requirements

---

### Pre-prepared scripts

In the ```scripts/``` folder, we provide pre-constructed commands for Llama 3.1 70B and Llama 3.1 405B distillation. 

**Llama 70B.** This model generally fits within an 8 $\times$ 80GB single node setup. We provide example scripts for launching the end-to-end LoLCATS procedure for the 70B model at: 
```
bash 
cd lolcats/
bash scripts/llama3_1_70b/70b_e2e_xent0_mse1000.sh
```

**Llama 405B.** We provide two strategies for distilling Llama 405B. 

The first approach uses the default LoLCATS and uses multi-node training to fit the model in memory. Here, each stage (attention transfer, LoRA finetuning) gets performed all at once. The second appraoch performs the attention transfer to the model *using smaller chunks* of the overall $126$ Transformer-layer Llama model. 

A sample end-to-end script (approach 1) is:
```
bash 
cd lolcats/
bash scripts/llama3_1_405b/405b_e2e_xent1_mse1000.sh
```

Sample scripts and details for the chunk-level approach (approach 2) are at: [this README.md](https://github.com/HazyResearch/lolcats/tree/lolcats-scaled/scripts/llama3_1_405b/trenchcoat)

---

### Running evaluations and inference.


---

RedPajama data: [https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora#data]
