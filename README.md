# LoLCATs

[WIP]

## Example Commands

```bash
python distill_llama.py \
--model_config distill_long_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_long_alpaca_8k_xent0_mse1000_lr1e-2_bs1 \
--finetune_config finetune_long_lora_qkvo_alpaca_clean_8192 \
--eval_config eval_alpaca_clean  \
--lk_zero_init --verbose --seed 0 --replicate 614 --state_chunk_len 1024 \
--num_train_epochs 2
```

cmd 3

```bash
python distill_llama.py \
--model_config distill_llama3_8b_lk_smd_wtk64_fd64_w01 \
--distill_config distill_alpaca_clean_xent0_mse1000_lr1e-2 \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean  \
--lk_zero_init --verbose --seed 0 --replicate 614 \
--num_train_epochs 2
```
