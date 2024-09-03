# LoLCATs [wip]

<p align="center">
<img src="assets/lolcats_and_tk_llamas.png" align='center' width=80% height=80%>
</p>

In this README:

- Getting started with dependencies, installation, and experiment configs
- Sample commands (Mistral-7B-v0.1, Llama-3-8B, Llama-3.1-8B, Llama-3.1-70B)

---

## Getting started

### Setup dependencies

Please see `environment.yaml` for dependencies. We can set them up with conda:

```
conda env create -f environment.yaml
conda activate lolcats
```

---

### Experiment and model configs

We organize things under experiment and model config files (`.yaml`) in `./configs`.

- Files under `./configs/experiments/` to determine dataset, training hyperparameters (distillation / conversion; finetuning).
- Files under `./configs/models/` determine model setup.

For models, our scripts should automatically download the models from Hugging Face, but you should change the `cache_dir` to reflect where you want to save the weights.

For example:

```yaml
pretrained_config:
  pretrained_model_name_or_path: "mistralai/Mistral-7B-v0.1"
  cache_dir: "/scr-ssd/mzhang/models/mistral-7b-v0.1" # change this
  return_dict: true
  quantization: false
  device_map: auto
  low_cpu_mem_usage: true
  torch_dtype: bfloat16
  rope_theta: 10000.0
  attn_implementation: eager # if supervising with attention weights
```

---

### Additional dependencies

#### Causal linear attention CUDA kernel

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

To build the kernel (`causal_dot_product`), first check that the PyTorch CUDA version (e.g., as specified in the `environment.yaml`) matches that of your system.

Then activate the conda environment (`conda activate lolcats`), navigate to `./csrc/`, and run `python setup.py install` within `./csrc/`, i.e.,

```bash
conda activate lolcats
cd ./csrc/
python setup.py install
```

It's worth checking the arguments in `./csrc/setup.py` to match your GPU setup and C++ versions.

### ThunderKittens linear attention + sliding window kernel

TODO. [ThunderKittens](https://github.com/HazyResearch/ThunderKittens)

#### More

We're very excited to integrate additional developments like Songlin and friends' [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)

---

#### Flash Attention 2 install

To train subquadratic analogs with Flash Attention 2 (FA2), we recommend following Tri's default instructions [here](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features).

Copying those instructions here: (1) Have `packaging` installed (`pip install packaging`). (2) Have `ninja` installed and working correctly (`ninja --version` then `echo $?` should return exit code 0). Otherwise reinstall with `pip uninstall -y ninja && pip install ninja`. (3) Install FA2 with

```
pip install flash-attn --no-build-isolation
```

---

## Sample commands

For any of these commands, you may need to provide a Hugging Face token to download model checkpints. Simply add the `--huggingface_token <your-token-here>` argument to any script below.

### Demoing linear attention 7B models

**_Note: Stale_**

We upload a couple checkpoints in `./checkpoints/`, where for any linearized 7B model we only need to save the linear attention layers and the LoRA weights (in two separate `.pt` checkpoints). To chat with these models, you can run:

```
python -Wignore demo_lolcats_llm.py \
--attn_mlp_checkpoint_path './checkpoints/distill_mistral_7b_lk_smd_zi/dl-d=distill_alpaca_clean_mistral_lr1e-2-m=distill_mistral_7b_lk_smd_eins-f=finetune_lora_qkvo_alpaca_clean_mistral-s=0-se=0-re=31-lk=untied_head_einsum-lsc=1-lzi=1_distill.pt' \
--finetune_checkpoint_path './checkpoints/distill_mistral_7b_lk_smd_zi/dl-d=distill_alpaca_clean_mistral_lr1e-2-m=distill_mistral_7b_lk_smd_eins-f=finetune_lora_qkvo_alpaca_clean_mistral-s=0-se=0-re=31-lk=untied_head_einsum-lsc=1-lzi=1-bs=1-gas=8-nte=2-ms=-1-es=100-se=0-re=31_ft.pt' \
--num_generations 1 --benchmark
```

---

### Linearizing 7B models

<p align="center">
<img src="assets/hedgehog_llamas.png" align='center' width=30% height=30%>
</p>

Any of the below commands will convert a 7B Mistral or Llama LLM into a subquadratic attention instruction-following variant. Despite only using LoRA and training on these 50K instruction-tuning samples, we're able to ``unlock'' a good amount of the base model performance when measured on LM Eval tasks.

See `configs/model/` for model configs used in the below commands, and `configs/experiments/` for attention transfer and finetuning configs.

#### Mistral-7B-v0.1, Hedgehog Feature Map, using Alpaca-Clean

```
python distill_llama.py --model_config distill_mistral_7b_lk_smd_fd64 \
--distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean \
--lk_zero_init \
--verbose --seed 0 --replicate 0 \
--huggingface_token hf_<insert your token here>
```

#### Mistral-7B-v0.1, Hedgehog + ThunderKittens Sliding Window, using Alpaca-Clean

```
python distill_llama.py --model_config distill_mistral_7b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean \
--lk_zero_init \
--verbose --seed 0 --replicate 0 \
--huggingface_token hf_<insert your token here>
```

#### Llama 3 8B, Hedgehog Feature Map, using Alpaca-Clean

```
python distill_llama.py --model_config distill_llama3_8b_lk_smd_fd64 \
--distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean \
--lk_zero_init \
--verbose --seed 0 --replicate 0 \
--huggingface_token hf_<insert your token here>
```

#### Llama 3 8B, Hedgehog + ThunderKittens Sliding Window, using Alpaca-Clean

```
python distill_llama.py --model_config distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean \
--lk_zero_init \
--verbose --seed 0 --replicate 0 \
--huggingface_token hf_<insert your token here>
```

#### Llama 3.1 8B, Hedgehog Feature Map, using Alpaca-Clean

```
python distill_llama.py --model_config distill_llama3_1_8b_lk_smd_fd64 \
--distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean \
--lk_zero_init \
--verbose --seed 0 --replicate 0 \
--huggingface_token hf_<insert your token here>
```

#### Llama 3.1 8B, Hedgehog + ThunderKittens Sliding Window, using Alpaca-Clean

```
python distill_llama.py --model_config distill_llama3_1_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean \
--lk_zero_init \
--verbose --seed 0 --replicate 0 \
--huggingface_token hf_<insert your token here>
```

---

### Evaluation

The above scripts will save two checkpoints: (1) for the learned attention layer weights (denoted by a `_distill` suffix), (2) for the LoRA finetuning weights (denoted by a `_ft` suffix). To evaluate linearized models from these checkpoints, we can add the `--load_distill_checkpoint` and `--load_finetune_checkpoint` args. For example:

```
python distill_llama.py --model_config distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean \
--lk_zero_init \
--verbose --seed 0 --replicate 0 \
--load_distill_checkpoint <path-to-distill-checkpoint> \
--load_finetune_checkpoint <path-to-finetune-checkpoint>
```

#### LM Evaluation Harness

For sample LM Eval scripts, please see `./lm_eval_harness/README.md`. An example such script is:

```bash
python lm_eval_harness/eval_lm_harness.py \
--model_type lolcats_ckpt \
--attn_mlp_checkpoint_path ./checkpoints/distill_mistral_7b_lk_smd_wtk64_fd64_w01/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2-m=distill_mistral_7b_lk_smd_wtk64_fd64_w01-f=finetune_long_lora_qkvo_alpaca_clean-s=0-gas=8-nte=2-se=0-re=614-scl=1024-lzi=1_distill.pt \
--finetune_checkpoint_path ./checkpoints/distill_mistral_7b_lk_smd_wtk64_fd64_w01/dl-d=dacxmldm7lswfwfllqac082061_lzi=1_distill1d-m=distill_mistral_7b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean-s=0-gas=8-nte=2-se=0-re=614-scl=1024-lzi=1-gas=8-nte=2-se=0-re=614_ft.pt \
--task piqa --num_shots 0  --no_cache --verbose
```

To setup the evaluations, we clone the Language Model Evaluation Harness from [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/b281b0921b636bc36ad05c0b0b0763bd6dd43463) to a separate directory (e.g., outside the lolcats directory).

- Note we use the `b281b09` branch following Hugging Face's [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

We then point to this path in `./lm_eval_harness/eval_lm_harness.py`, e.g.

```python
LM_EVALUATION_HARNESS_PATH = '/juice2/scr2/mzhang/projects/lm-evaluation-harness'  # Change this to where you clone LM eval harness from
```

---

### Linearizing 70B models and up [WIP]

<p align="center">
<img src="assets/hedgehog_llamas_big.png" align='center' width=40% height=40%>
</p>

We also support linearizing larger LLMs (Llama 3.1 70B, Llama 3.1 405B) using the great [llama-recipes](https://github.com/meta-llama/llama-recipes/tree/main/src/llama_recipes) repository.

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

## Setup Debugging

### Huggingface datasets errors

If you come across an error like the following:

```
  File "/root/miniconda3/envs/hedgehog/lib/python3.12/site-packages/fsspec/spec.py", line 606, in glob
    pattern = glob_translate(path + ("/" if ends_with_sep else ""))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/hedgehog/lib/python3.12/site-packages/fsspec/utils.py", line 734, in glob_translate
    raise ValueError(
ValueError: Invalid pattern: '**' can only be an entire path component
```

Try reinstalling the Hugging Face `datasets` package with the version specified, e.g., via `pip install datasets==2.15.0`.

Sometimes setting up the virtual environment from `environment.yaml` results in `datasets==2.11.0` being installed instead.

Similarly, you may need to run the following installs:

```bash
pip install nltk
pip install rouge-score
```

### Miniconda installation

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

### `causal_dot_product` kernel installation

If running `python setup.py install` in `./csrc/` fails, try making sure your environment's CUDA version matches that of your system. In our case, specifying

```yaml
- pytorch-cuda=12.1
```

in `environment.yaml` for a system with CUDA 12.2 worked.

Also consider checking that your CUDA install is accessible, e.g., by adding the following to your `.bashrc`:

```
export CUDA_HOME=/usr/local/cuda-12.2/
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
```
