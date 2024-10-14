
We include configs for performing linearization with two different training datasets -- [Alpaca](https://huggingface.co/datasets/yahma/alpaca-cleaned) and [RedPajama](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora#data):


Alpaca:
- Distillation config: `distill_alpaca_llama_405b_xent0_mse1000_lr1e-2.yaml` *or* `distill_alpaca_llama_405b_xent1_mse1000_lr1e-2.yaml`, depending on if you want to include a cross entropy loss term between the softmax and linear attention $N^2$ **maps** during attention transfer in addition to the MSE loss on attention **outputs**.
- Finetune configs: `finetune_alpaca_llama_405b.yaml` *or* `finetune_alpaca_llama_405b_qkvo.yaml` *or* `finetune_alpaca_llama_405b_qkvo_e2.yaml`, depending on if you want to add LoRA to the q and v projection, or all qkvo attention projections and depending on the number of epochs you want to fine-tune for.

Red pajama:
- Distillation config: `rp_distill_llama_405b_xent0_mse1000_lr1e-2.yaml` *or* `rp_distill_llama_405b_xent1_mse1000_lr1e-2.yaml`, depending on if you want to include a cross entropy loss term between the softmax and linear attention $N^2$ **maps** during attention transfer in addition to the MSE loss on attention **outputs**.
- Finetune configs: `rp_finetune_llama_40b_qv_hparams.yaml` *or* `finetune_rp_llama_405b_qkvo_e2.yaml`, depending on if you want to add LoRA to the q and v projection, or all qkvo attention projections and depending on the number of epochs you want to fine-tune for.

Sample scripts for linearization, using these configs, are provided at: `lolcats/scripts/llama3_1_405b/`

