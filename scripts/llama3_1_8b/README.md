
We provide sample scripts to linearize Llama 3.1 8b. We include scripts for 2 experiments in this folder, and you can build off these samples for your own experiments. Any of these experiments can be run on a single 80GB H100 GPU.

*Note* In each script and config, make sure any required paths (e.g., for models, data) are updated to how you have set up your environments.

### LoLCATS Linearization 

1. Alpaca data, 1024 sequence length
```
bash
cd lolcats/
bash scripts/llama3_1_8b/alpaca_data/standard_alpaca_baseline.sh
```

And run eval with LM-Eval harness:
```
bash
cd lolcats/
bash scripts/llama3_1_8b/alpaca_data/eval.sh
```

2. RedPajama data, 1024 sequence length
```
bash
cd lolcats/
bash scripts/llama3_1_8b/redpajama_data/standard_rp.sh
```

And run eval with LM-Eval harness: 
```
bash
cd lolcats/
bash scripts/llama3_1_8b/redpajama_data/eval.sh
```

