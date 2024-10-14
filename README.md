
# LoLCATS in a trenchcoat

<p align="center">
<img src="assets/hedgehog_llamas_big.png" align='center' width=35% height=35%>
</p>

This readme describes how to obtain scaled up, 70B and 405B parameter, linear attention LLMs! 

Please see the main branch README for setup and installation instructions. Follow these instructions to install the RedPajama data to your local machine: [Long LLM Repo](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora#data)


## Overview: Linearizing the Llama 3.1 family.

This section provides sample scripts with commands to train your own models. As a brief overview, the commands have a (1) *distill-stage config* and (2) *finetune-stage config* to specify the optimizer scheduler and data for the attention transfer and LoRA fine-tune stages respectively. The commands also have a (3) model config to specify the architecture. 

### Linearizing Llama 8B

We provide a scripts to linearize Llama 3.1 8B, and this can be completed using a single 80GB H100 GPU, within <1 day. Note that the main branch of this repo focuses on 8B parameter models and uses the same structure.

*Code structure*: The main loop for linearization is at [distill_llama.py](https://github.com/HazyResearch/lolcats/blob/lolcats-scaled/distill_llama.py).

*Scripts*: Find the scripts and discussion at [this README.md](https://github.com/HazyResearch/lolcats/tree/lolcats-scaled/scripts/llama3_1_8b).


### Linearizing Llama 70B

We provide a scripts to linearize Llama 3.1 70B, and this can be completed on a single 8x80GB H100 GPU node, within <1 day.

*Code structure*: We use two files located at:
1. `llama-recipes/distill_llama.py`: "attention transfer", where we first train subquadratic attentions to mimic the softmax attentions (saving the learned attention feature map checkpoints)
2. `llama-recipes/distill_llama_finetune.py`: "LoRA finetuning", where we swap in the learned attentions and finetune the rest of the model with LoRA (saving the LoRA checkpoints)
By passing in the same configurations files and arguments to both scripts, `distill_llama_finetune.py` should automatically load the saved checkpoints and pick up from where `distill_llama.py` left off.


*Scripts*: We provide example scripts for launching the end-to-end LoLCATS procedure for the 70B model at [this README.md](https://github.com/HazyResearch/lolcats/tree/lolcats-scaled/scripts/llama3_1_70b).


### Linearizing Llama 405B

We provide a scripts to linearize Llama 3.1 405B, and this can be completed with access to three nodes of 8x80GB H100 GPUs, within a few days.

*Code structure*: For 405B, our general approach is to (1) break the Llamas into little blocks (["crias", baby llamas](https://en.wikipedia.org/wiki/Cria)) of layers, (2) replace the softmax-attentions with LoLCATS linear attentions and train the linear layers' feature maps to match the outputs from softmax, (3) put all the little LoLCATS back together with some LoRA adaptation -- hence, LoLCATS in a Trenchcoat. 

Please find more discussion on this block-wise approach and our motivation for it (as opposed to joint attention transfer across all layers) in Section 3 of our paper. 

The core files for trenchcoat training are located at [lolcats/llama_recipes/trenchcoat_lolcat](https://github.com/HazyResearch/lolcats/tree/lolcats-scaled/llama_recipes/trenchcoat_lolcat):

1. [shard_llama_to_cria.py](https://github.com/HazyResearch/lolcats/blob/lolcats-scaled/llama_recipes/trenchcoat_lolcat/shard_llama_to_cria.py): splits the 126 Llama 3.1 layers into blocks of $k$ layers and saves these blocks in .pt files. This takes <30 minutes.

2. [save_llama_attn_inputs.py](https://github.com/HazyResearch/lolcats/blob/lolcats-scaled/llama_recipes/trenchcoat_lolcat/save_llama_attn_inputs.py): saves the hidden states from block $i$, for all $i$, to your hard disk. The states from block $i$ are the training inputs for block $i+1$! This takes <1 hour, make sure you have a Terrabytes of disk space available.

3. [distill_mini_train.py](https://github.com/HazyResearch/lolcats/blob/lolcats-scaled/llama_recipes/trenchcoat_lolcat/distill_mini_train.py): the core file for performing attention-transfer between softmax Llama 3.1 attentions and our linear attention layers in LoLCATS. Here, each block can typically be trained on a *single* GPU (e.g., at $k=9$, sequence length $1024$).

4. [stitch_mini_finetune.py](https://github.com/HazyResearch/lolcats/blob/lolcats-scaled/llama_recipes/trenchcoat_lolcat/stitch_mini_finetune.py): stitch the blocks together and fine-tune them with LoRA. Here, we need multiple nodes to fit the full stitched Llama 3.1 405B model -- we're back at 126 layers.

5. [save_fsdp_to_pt.py](https://github.com/HazyResearch/lolcats/blob/lolcats-scaled/llama_recipes/trenchcoat_lolcat/save_fsdp_to_pt.py): take all the sharded files from the LoRA stage FSDP and put them in a single .pt file for convenience. This takes a couple of seconds.


*Scripts* We provide sample scripts and details for the block-by-block approach (approach 2) at: [this README.md](https://github.com/HazyResearch/lolcats/tree/lolcats-scaled/scripts/llama3_1_405b/data=rp_len=1024_trenchcoat/) We also discuss and point to sample scripts for some of the baseline approaches to block-wise 405B at that README.

**IMPORTANT: Additional Setup Note**: To use the cria-by-cria distillation approach: we need to set "self.register_buffer("inv_freq", inv_freq, persistent=True) in the [Transformers modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) file, when you install.
```
inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
self.register_buffer("inv_freq", inv_freq, persistent=True) # LoLCATS Flag 
self.original_inv_freq = self.inv_freq
```

## Quick demos and inference

We provide details at [this README.md](https://github.com/HazyResearch/lolcats/tree/lolcats-scaled/inference).


Feel free to reach out if you have questions!


