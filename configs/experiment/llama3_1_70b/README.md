
We include configs for performing linearization with two different training datasets -- [Alpaca](https://huggingface.co/datasets/yahma/alpaca-cleaned) and [RedPajama](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora#data):


Alpaca:
- Distillation config: `distill_alpaca_llama_70b_xent0_mse1000_lr1e-2.yaml`
- Finetune configs: `finetune_alpaca_llama_70b_qv.yaml` *or* `finetune_alpaca_llama_70b_qkvo.yaml`, depending on if you want to add LoRA to the q and v projection, or all qkvo attention projections.

Red pajama:
- Distillation config: `distill_rp_llama_70b_xent0_mse1000_lr1e-2.yaml`
- Finetune configs: `finetune_rp_llama_70b_qv.yaml` *or* `finetune_rp_llama_70b_qkvo.yaml`, depending on if you want to add LoRA to the q and v projection, or all qkvo attention projections.

Red pajama at 2048 sequence length (the default length is 1024): 
- Distillation config: `distill_rp2048_llama_70b_xent0_mse1000_lr1e-2.yaml`
- Finetune configs: `finetune_rp2048_llama70b_qv.yaml`


Sample scripts for linearization, using these configs, are provided at: `lolcats/scripts/llama3_1_8b/`

