# LM Evaluation Harness Setup + Sample Scripts

To setup the evaluations, we clone the Language Model Evaluation Harness from [here](https://github.com/EleutherAI/lm-evaluation-harness/tree/b281b0921b636bc36ad05c0b0b0763bd6dd43463) to a separate directory (e.g., outside the lolcats directory).

- Note we use the `b281b09` branch following Hugging Face's [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

```bash
git checkout b281b09
```

We then point to this path in `./lm_eval_harness/eval_lm_harness.py`, e.g.

```python
LM_EVALUATION_HARNESS_PATH = '/juice2/scr2/mzhang/projects/lm-evaluation-harness'  # Change this to where you clone LM eval harness from
```

You may also need to install the following packages:

```bash
pip install --upgrade --force-reinstall sacrebleu
pip install evaluate sqlitedict scikit-learn pycountry
```

Finally, we'll want to replace the current file in `lm-evaluation-harness/lm_eval/models/huggingface.py` with `lolcats/lm_eval_harness/models_huggingface.py` to better support loading our linearized checkpoints (some missing keyword args in the original... sorry).
---

## Running the evaluations

All evaluation scripts take the following template:

```bash
python lm_eval_harness/eval_lm_harness.py \
--model_type lolcats_ckpt \
--attn_mlp_checkpoint_path <path-to-attention-feature_map-checkpoints> \
--finetune_checkpoint_path <path-to-post-swap-lora-checkpoints> \
--task <task-name> --num_shots <num-shots> --no_cache --verbose
```

We provide examples of such below.

---

### PiQA (zero-shot)

```bash
python lm_eval_harness/eval_lm_harness.py \
--model_type lolcats_ckpt \
--attn_mlp_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_long_lora_qkvo_alpaca_clean_8192-s=0-gas=1-nte=2-se=0-re=614-scl=1024-lzi=1_distill.pt \
--finetune_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_long_lora_qkvo_alpaca_clean_8192-s=0-gas=1-nte=2-se=0-re=614-scl=1024-lzi=1-bs=1-gas=1-nte=2-se=0-re=614_ft.pt \
--task piqa --num_shots 0 --no_cache --verbose

```

### ARC-Easy (zero-shot)

```bash
python lm_eval_harness/eval_lm_harness.py \
--model_type lolcats_ckpt \
--attn_mlp_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_long_lora_qkvo_alpaca_clean_8192-s=0-gas=1-nte=2-se=0-re=614-scl=1024-lzi=1_distill.pt \
--finetune_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_long_lora_qkvo_alpaca_clean_8192-s=0-gas=1-nte=2-se=0-re=614-scl=1024-lzi=1-bs=1-gas=1-nte=2-se=0-re=614_ft.pt \
--task arc_easy --num_shots 0 --no_cache --verbose
```

### ARC-Challenge (zero-shot)

```bash
python lm_eval_harness/eval_lm_harness.py \
--model_type lolcats_ckpt \
--attn_mlp_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_long_lora_qkvo_alpaca_clean_8192-s=0-gas=1-nte=2-se=0-re=614-scl=1024-lzi=1_distill.pt \
--finetune_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_long_lora_qkvo_alpaca_clean_8192-s=0-gas=1-nte=2-se=0-re=614-scl=1024-lzi=1-bs=1-gas=1-nte=2-se=0-re=614_ft.pt \
--task arc_challenge --num_shots 0 --no_cache --verbose
```

### Hellaswag (zero-shot)

```bash
python lm_eval_harness/eval_lm_harness.py --model_type lolcats_ckpt \
--attn_mlp_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_long_lora_qkvo_alpaca_clean_8192-s=0-gas=1-nte=2-se=0-re=614-scl=1024-lzi=1_distill.pt \
--finetune_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_long_lora_qkvo_alpaca_clean_8192-s=0-gas=1-nte=2-se=0-re=614-scl=1024-lzi=1-bs=1-gas=1-nte=2-se=0-re=614_ft.pt \
--task hellaswag --num_shots 0 --no_cache --verbose
```

### Winogrande (zero-shot)

```bash
python lm_eval_harness/eval_lm_harness.py --model_type lolcats_ckpt \
--attn_mlp_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_long_lora_qkvo_alpaca_clean_8192-s=0-gas=1-nte=2-se=0-re=614-scl=1024-lzi=1_distill.pt \
--finetune_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_long_lora_qkvo_alpaca_clean_8192-s=0-gas=1-nte=2-se=0-re=614-scl=1024-lzi=1-bs=1-gas=1-nte=2-se=0-re=614_ft.pt \
--task winogrande --num_shots 0 --no_cache --verbose
```

### MMLU (5-shot)

```bash
python lm_eval_harness/eval_lm_harness.py --model_type lolcats_ckpt \
--attn_mlp_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_long_lora_qkvo_alpaca_clean_8192-s=0-gas=1-nte=2-se=0-re=614-scl=1024-lzi=1_distill.pt \
--finetune_checkpoint_path ./checkpoints/distill_long_llama3_8b_lk_smd_wtk64_fd64_w01/dl-d=distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1-m=distill_long_llama3_8b_lk_smd_wtk64_fd64_w01-f=finetune_long_lora_qkvo_alpaca_clean_8192-s=0-gas=1-nte=2-se=0-re=614-scl=1024-lzi=1-bs=1-gas=1-nte=2-se=0-re=614_ft.pt \
--task hendrycksTest --num_shots 5 --no_cache --verbose
```

### 70B Evaluation

For running 70B evals, we can use the following example

```bash
python lm_eval_harness/eval_lm_harness_big.py \
--model_config distill_llama3_1_70b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_llama3_1_70b_xent1_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean_llama3_1_70b \
--lk_zero_init \
--verbose --replicate 0 --seed 0 \
--eval_steps 10 --dataset_chunk_size 256 \
--enable_fsdp --low_cpu_fsdp \
--task piqa --num_shots 0 --no_cache \
--attn_mlp_checkpoint_path /scr-ssd/mzhang/projects/lolcats/checkpoints/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01/distill-dl-d=distill_alpaca_clean_llama3_1_70b_xent1_mse1000_lr1e-2-m=distill_llama3_1_70b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean_llama3_1_70b-dcs=512-se=0-re=0-at=lolcats_llama_window_tk_bf16 \
--finetune_checkpoint_path /scr-ssd/mzhang/projects/lolcats/checkpoints/distill_llama3_1_70b_lk_smd_wtk64_fd64_w01/finetune-dl-d=distill_alpaca_clean_llama3_1_70b_xent1_mse1000_lr1e-2-m=distill_llama3_1_70b_lk_smd_wtk64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean_llama3_1_70b-dcs=256-se=0-re=0-at=lolcats_llama_window_tk_bf16-dcs=256-se=0-re=0
```

where `attn_mlp_checkpoint_path` and `finetune_checkpoint_path` now point to the directory paths where the sharded attention and finetune checkpoints are saved.
