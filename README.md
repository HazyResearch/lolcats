
# LoLCATS in a trenchcoat

<p align="center">
<img src="assets/hedgehog_llamas_big.png" align='center' width=35% height=35%>
</p>

This readme describes how to obtain scaled up, 70B and 405B parameter, linear attention LLMs! Our general approach is to (1) break the Llamas into little blocks (["crias", baby llamas](https://en.wikipedia.org/wiki/Cria)) of layers, (2) replace the softmax-attentions with LoLCATS linear attentions and train the linear layers' feature maps to match the outputs from softmax, (3) put all the little LoLCATS back together with some LoRA adaptation -- hence, LoLCATS in a Trenchcoat.


Please see the main branch README for setup and installation instructions. Follow these instructions to install the RedPajama data: [Long LLM Repo](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora#data)


## Overview: Linearizing large language models.

#### Linearizing Llama 70B

First, we describe how to linearize at 70B parameters, which is a good starting point. We split the two stages of LoLCATs linearizing into two scripts:

1. `llama-recipes/distill_llama.py`: "attention transfer", where we first train subquadratic attentions to mimic the softmax attentions (saving the learned attention feature map checkpoints)
2. `llama-recipes/distill_llama_finetune.py`: "LoRA finetuning", where we swap in the learned attentions and finetune the rest of the model with LoRA (saving the LoRA checkpoints)

By passing in the same configurations files and arguments to both scripts, `distill_llama_finetune.py` should automatically load the saved checkpoints and pick up from where `distill_llama.py` left off.

We perform this distillation on a single 8x80GB GPU node, for reference. It takes <1 day.

**Llama 70B.** We provide example scripts for launching the end-to-end LoLCATS procedure for the 70B model at: 
```
bash 
cd lolcats/
bash scripts/llama3_1_70b/70b_e2e_xent0_mse1000.sh
```
As a brief overview, we have a distill-stage config and finetune-stage config to specify the optimizer scheduler and data, and a model config to specify the architecture, in these commands.

#### Linearizing Llama 405B

For 405B, we use the trenchcoats block-by-block approach specified above. Sample scripts and details for the block-by-block approach (approach 2) are at: [this README.md](https://github.com/HazyResearch/lolcats/tree/lolcats-scaled/scripts/llama3_1_405b/trenchcoat)

*Warning*: To use the cria-by-cria distillation approach: we need to set "self.register_buffer("inv_freq", inv_freq, persistent=True) in the [Transformers modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) file, when you install.

```
inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
self.register_buffer("inv_freq", inv_freq, persistent=True) # LoLCATS Flag 
self.original_inv_freq = self.inv_freq
```

Baseline to our approach: As a reference-point for the value of the block-by-block, we also provide a script to train a baseline that does attention-transfer without block by block and then LoRA.
```
bash 
cd lolcats/
bash scripts/llama3_1_405b/405b_e2e_xent1_mse1000.sh
```

## Evals and inference

We provide evaluation code at [this README.md](https://github.com/HazyResearch/lolcats/tree/lolcats-scaled/inference).

