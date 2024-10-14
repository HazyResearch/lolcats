
We provide sample scripts to linearize Llama 3.1 405b. You can build off these samples for your own experiments. Please see our paper for more discussion on joint versus block-wise attention transfer. At this scale, both should be good to use. Any of these experiments can be run on 3 8x80GB H100 nodes. We provide slurm scripts for distributed multi-node.

*Note* In each script and config, make sure any required paths (e.g., for models, data) are updated to how you have set up your environments.

### LoLCATS Linearization 

1. RedPajama data, 1024 sequence length, joint attention transfer over layers. **As discussed in our paper, the joint transfer is difficult at the 405B scale. Block-wise transfer, as shown in the next scripts, leads to better quality.**
```
bash
cd lolcats/
sbatch scripts/llama3_1_405b/405b_e2e_lr1e-2_n1024.sh
```


2. RedPajama data, 1024 sequence length, block-wise attention transfer over 14 blocks with 9 layers per block

First, shard Llama 3.1 405B into blocks of 9 layers each:
```
bash
cd lolcats/
sbatch scripts/llama3_1_405b/data=rp_len=1024_trenchcoat/cria_prepare/shard.sh
```

Second, collect the hidden states after running data through each block of 9 layers and save them to disk.
```
bash
cd lolcats/
sbatch scripts/llama3_1_405b/data=rp_len=1024_trenchcoat/cria_prepare/collect_inputs.sh
```

Third, run attention transfer on each of the 14 blocks of 9 layers (each script can run on a *single* GPU):
```
bash
cd lolcats/
sbatch scripts/llama3_1_405b/data=rp_len=1024_trenchcoat/cria_train/1.sh
...
sbatch scripts/llama3_1_405b/data=rp_len=1024_trenchcoat/cria_train/14.sh
```

Fourth, stitch the blocks together with LoRA finetuning:
```
bash
cd lolcats/
sbatch scripts/llama3_1_405b/data=rp_len=1024_trenchcoat/cria_prepare/finetune.sh

cd lolcats/
sbatch scripts/llama3_1_405b/data=rp_len=1024_trenchcoat/cria_prepare/stitch.sh
```

For evaluation with the eval harness, you can adapt the script:
```
bash
cd lolcats/
sbatch scripts/llama3_1_405b/data=rp_len=1024_trenchcoat/405b_mmlu_eval.sh
```


### Baseline approach

We also provide code to reproduce the results of *only* using LoRA fine-tuning with *no attention transfer*. This is similar to prior works on linearization, which do not use attention transfer. 
```
bash
cd lolcats/
sbatch scripts/llama3_1_405b/baselines/405b_no_distill_baseline.sh
```
