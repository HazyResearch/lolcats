# LoLCATs [wip]
<p align="center">
<img src="lolcats_and_tk_llamas.png" align='center' width=80% height=80%>
</p>

In this README:  
* Getting started with dependencies, installation, and experiment configs  
* Sample commands (Mistral-7B-v0.1, Llama-2-7B, Llama-3-8B, Mixtral-8x7B, Llama-2-70B, Llama-3-70B)

---

## Getting started

### Setup dependencies
Please see `environment.yaml` for dependencies. We can set them up with conda:
```
conda env create -f environment.yaml
conda activate hedgehog
```

---

### Experiment and model configs
We organize things under experiment and model config files (`.yaml`) in `./configs`.
*  Files under `./configs/experiments/` to determine dataset, training hyperparameters (distillation / conversion; finetuning).
*  Files under `./configs/models/` determine model setup.

For models, our scripts should automatically download the models from Hugging Face, but you should change the `cache_dir` to reflect where you want to save the weights. 

For example:
```yaml
pretrained_config:
  pretrained_model_name_or_path: 'mistralai/Mistral-7B-v0.1'
  cache_dir: '/scr-ssd/mzhang/models/mistral-7b-v0.1'  # change this
  return_dict: true
  quantization: false
  device_map: auto
  low_cpu_mem_usage: true
  torch_dtype: bfloat16
  rope_theta: 10000.0
  attn_implementation: eager  # so we can supervise with attention weights
```

---

### Causal linear attention CUDA kernel
For now, we implement the causal linear attention with the CUDA kernel from [https://github.com/idiap/fast-transformers/tree/master](https://github.com/idiap/fast-transformers/tree/master), citing:  
```
@inproceedings{katharopoulos_et_al_2020,
    author = {Katharopoulos, A. and Vyas, A. and Pappas, N. and Fleuret, F.},
    title = {Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention},
    booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
    year = {2020}
}

@article{vyas_et_al_2020,
    author={Vyas, A. and Katharopoulos, A. and Fleuret, F.},
    title={Fast Transformers with Clustered Attention},
    booktitle = {Proceedings of the International Conference on Neural Information Processing Systems (NeurIPS)},
    year={2020}
}
```
To build the kernel (`causal_dot_product`), first activate the conda environment (`conda activate hedgehog`). Then navigate to `./csrc/` and run `python setup.py install` within `./csrc/`. It's worth checking the arguments in `./csrc/setup.py` to match your GPU setup and C++ versions.

TODO: we're very excited to integrate additional developments like Songlin and friends' `flash-linear-attention` [repo](https://github.com/sustcsonglin/flash-linear-attention), as well as [ThunderKittens](https://github.com/HazyResearch/ThunderKittens). Please let us know if you're interested in scaling up these efficient linear attention implementations into 7B to 70B models.

### Flash Attention 2 install
While we don't use Flash Attention 2 (FA2), it can be helpful to have this installed for benchmarking comparisons or other distillation pipelines (e.g., finetune LLM with FA2 first on custom data, then distill it into a linear attention variant with the trained attentions). 

We recommend following Tri's default instructions [here](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features). 

Copying those instructions here: (1) Have `packaging` installed (`pip install packaging`). (2) Have `ninja` installed and working correctly (`ninja --version` then `echo $?` should return exit code 0). Otherwise reinstall with `pip uninstall -y ninja && pip install ninja`. (3) Install FA2 with
```
pip install flash-attn --no-build-isolation
```

---

## Sample commands

For any of these commands, you may need to provide a Hugging Face token to download model checkpints. Simply add the `--huggingface_token <your-token-here>` argument to any script below.  

### Demoing linear attention 7B models  
We upload a couple checkpoints in `./checkpoints/`, where for any linearized 7B model we only need to save the linear attention layers and the LoRA weights (in two separate `.pt` checkpoints). To chat with these models, you can run:

```
python -Wignore demo_hedgehog_llm.py \
--attn_checkpoint_path './checkpoints/distill_mistral_7b_lk_smd_zi/dl-d=distill_alpaca_clean_mistral_lr1e-2-m=distill_mistral_7b_lk_smd_eins-f=finetune_lora_qkvo_alpaca_clean_mistral-s=0-se=0-re=31-lk=untied_head_einsum-lsc=1-lzi=1_distill.pt' \
--peft_checkpoint_path './checkpoints/distill_mistral_7b_lk_smd_zi/dl-d=distill_alpaca_clean_mistral_lr1e-2-m=distill_mistral_7b_lk_smd_eins-f=finetune_lora_qkvo_alpaca_clean_mistral-s=0-se=0-re=31-lk=untied_head_einsum-lsc=1-lzi=1-bs=1-gas=8-nte=2-ms=-1-es=100-se=0-re=31_ft.pt' \
--num_generations 1 --benchmark
```


### Distilling + finetuning 7B models  
Any of the below commands will convert a 7B Mistral or Llama LLM into a linear attention instruction-following variant. Despite only using LoRA and training on these 50K instruction-tuning samples, we're able to ``unlock'' a good amount of the base model performance when measured on LM Eval tasks.  

#### Mistral-7B-v0.1, Hedgehog Feature Map, using Alpaca-Clean
```
python distill_llama.py --model_config distill_mistral_7b_lk_smd_zi \
--distill_config distill_alpaca_clean_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean \
--lk_skip_connection --lk_zero_init \
--verbose --seed 0 --replicate 0 
```

#### Mistral-7B-v0.1, Hedgehog + Sliding Window, using Alpaca-Clean
```
python distill_llama.py --model_config distill_mistral_7b_lk_smd_zi_swa16_hh \
--distill_config distill_alpaca_clean_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean \
--lk_skip_connection --lk_zero_init \
--verbose --seed 0 --replicate 0 
```

#### Llama-3-8B, Hedgehog Feature Map, using Alpaca-Clean
```
python distill_llama.py --model_config distill_llama3_8b_lk_smd_zi \
--distill_config distill_alpaca_clean_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean \
--lk_skip_connection --lk_zero_init \
--verbose --seed 0 --replicate 0 \
--huggingface_token hf_<insert your token here>
```

#### Llama-2-7B, Hedgehog Feature Map, using Alpaca-Clean
```
python distill_llama.py --model_config distill_llama2_7b_lk_smd_zi \
--distill_config distill_alpaca_clean_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean \
--lk_skip_connection --lk_zero_init \
--verbose --seed 0 --replicate 0 \
--huggingface_token hf_<insert your token here>
```

### Evaluation  

The above scripts will save two checkpoints: (1) for the learned attention layer weights (denoted by a `_distill` suffix), (2) for the LoRA finetuning weights (denoted by a `_ft` suffix). To evaluate linearized models from these checkpoints, we can add the `--load_distill_checkpoint` and `--load_finetune_checkpoint` args. For example:

```
python distill_llama.py --model_config distill_mistral_7b_lk_smd_zi_swa16_hh \
--distill_config distill_alpaca_clean_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean \
--lk_skip_connection --lk_zero_init \
--verbose --seed 0 --replicate 0 \
--load_distill_checkpoint ./checkpoints/distill_mistral_7b_lk_smd_zi_swa16_hh/dl-d=distill_alpaca_clean_kld_mse_mistral_lr1e-3-m=distill_mistral_7b_lk_smd_swa_hh-f=finetune_lora_qkvo_alpaca_clean_mistral-s=0-se=0-re=0-lk=untied_head_einsum-lsc=1_distill.pt \
--load_finetune_checkpoint ./checkpoints/distill_mistral_7b_lk_smd_zi_swa16_hh/dl-d=distill_alpaca_clean_kld_mse_mistral_lr1e-3-m=distill_mistral_7b_lk_smd_swa_hh-f=finetune_lora_qkvo_alpaca_clean_mistral-s=0-se=0-re=0-lk=untied_head_einsum-lsc=1-se=0-re=0_ft.pt
```

#### LM Eval 

For sample LM Eval scripts, please see `./lm_eval_harness/README.md`. In particular, this will involve cloning the Language Model Evaluation Harness from [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/b281b0921b636bc36ad05c0b0b0763bd6dd43463). Note we use the `b281b09` branch following Hugging Face's [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). 

---

## Updated example commands

```bash
python distill_llama.py \
--model_config distill_long_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1 \
--finetune_config finetune_long_lora_qkvo_alpaca_clean_8192 \
--eval_config eval_alpaca_clean  \
--lk_zero_init --verbose --seed 0 --replicate 614 --state_chunk_len 1024 \
--num_train_epochs 2
```

```bash
python distill_llama.py \
--model_config distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean  \
--lk_zero_init --verbose --seed 0 --replicate 614 \
--num_train_epochs 2
```
