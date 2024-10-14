
We provide sample scripts to linearize Llama 3.1 70b. We include scripts for 3 experiments in this folder, and you can build off these samples for your own experiments. Please see our paper for more discussion on joint versus block-wise attention transfer. At this scale, both should be good to use. Any of these experiments can be run on a single 8x80GB H100 node.

*Note* In each script and config, make sure any required paths (e.g., for models, data) are updated to how you have set up your environments.

### LoLCATS Linearization 

1. Alpaca data, 1024 sequence length, joint attention transfer over layers
```
bash
cd lolcats/
bash scripts/llama3_1_70b/70b_e2e_alpaca_xent0_mse1000.sh
```

2. RedPajama data, 1024 sequence length, joint attention transfer over layers
```
bash
cd lolcats/
bash scripts/llama3_1_70b/70b_e2e_rp_xent0_mse1000.sh
```

3. RedPajama data, 2048 sequence length, block-wise attention transfer over layers with 5 layers per block
First, shard Llama 3.1 70B into blocks of 5 layers each:
```
bash
cd lolcats/
bash scripts/llama3_1_70b/data\=rpcontig_len\=2048_trenchcoat/cria_prepare/shard.sh
```

Second, collect the hidden states after running data through each block of 5 layers and save them to disk.
```
bash
cd lolcats/
bash scripts/llama3_1_70b/data\=rpcontig_len\=2048_trenchcoat/cria_prepare/collect_inputs.sh
```

Third, run attention transfer on each of the 16 blocks of 5 layers (each script can run on a *single* GPU):
```
bash
cd lolcats/
bash scripts/llama3_1_70b/data\=rpcontig_len\=2048_trenchcoat/cria_train/1.sh
...
bash scripts/llama3_1_70b/data\=rpcontig_len\=2048_trenchcoat/cria_train/16.sh
```

Fourth, stitch the blocks together with LoRA finetuning. At the end, you should have a single `.pt` file that contains the linear attention learned feature maps and the LoRA weights etc. of LoLCATS.
```
bash
cd lolcats/
bash scripts/llama3_1_70b/data\=rpcontig_len\=2048_trenchcoat/cria_prepare/finetune.sh
```

### Baseline approach

We also provide code to reproduce the results of *only* using LoRA fine-tuning with *no attention transfer*. This is similar to prior works on linearization, which do not use attention transfer. 
```
bash
cd lolcats/
bash scripts/llama3_1_70b/baselines/70b_no_distill_baseline.sh
```

### Evaluation 

We provide code to run evaluation with either the LM-Eval Harness or a standalone MMLU evaluation *with no dependencies*:

For evaluation with the eval harness, you can adapt the script:
```
bash
cd lolcats/
bash scripts/llama3_1_70b/evaluate_models/run_lm_eval_harness.sh
```

For standalone 5-shot MMLU, you can adapt the script:
```
bash
cd lolcats/
bash scripts/llama3_1_70b/evaluate_models/run_mmlu_standalone.sh
```

