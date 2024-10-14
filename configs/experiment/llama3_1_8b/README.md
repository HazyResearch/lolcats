
We include configs for performing linearization with two different training datasets -- [Alpaca](https://huggingface.co/datasets/yahma/alpaca-cleaned) and [RedPajama](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora#data):

Alpaca:
- Distillation config: `distill_alpaca_clean_xent0_mse1000_lr1e-2.yaml`
- Finetune config: `finetune_qkvo_alpaca_clean.yaml`

Red pajama:
- Distillation config: `distill_rp_clean_xent0_mse1000_lr1e-2.yaml`
- Finetune configs: `finetune_qkvo_rp_clean.yaml`


Sample scripts for linearization, using these configs, are provided at: `lolcats/scripts/llama3_1_70b/`
