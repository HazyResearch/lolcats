
## Demos

We describe how to use LoLCATS checkpoints. We include:
1. Demo script to talk to our models using Hugging Face checkpoints
2. Demo script to benchmark the pretrained 8B linearized versus base softmax attention models
3. Code to reproduce the MMLU numbers at 70B and 405B numbers using our uploaded HuggingFace checkpoints
4. Coming soon: VLLM integration with custom LoLCATS CUDA kernels!

### Talk to pre-trained LoLCATS LLMs

Use the commands provided at `demo_8b.sh` to run inference with our LoLCATS - Llama 3.1 8B checkpoint, which will be downloaded from Hugging Face.  The downloaded checkpoints require under <1GB, and are inserted into your local Meta Llama 3.1 model in 16-bit precision -- please ensure you have downloaded the base model and specify your path to it in the configs in `demo_8b.sh`. To run the demo:
```bash
bash demo_8b.sh
```

### Fast inference with custom CUDA kernels

We provide a custom CUDA prefill kernel written in the [ThunderKittens framework](https://github.com/HazyResearch/ThunderKittens).

As a quick end-to-end compare the prefill speed of the linearized LoLCATS 8B vs. the base Llama 8B model, we provide a script at:
```bash
bash benchmark_8b.sh
```

The code will print out the inference tokens per second per method. 

### 5-shot MMLU Eval

First get the 5-shot MMLU data. We directly saved the tokenized examples produced by the [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) codebase to a pickle file
```
cd lolcats/inference/
unzip mmlu.pkl.zip
```

We used these scripts, plugging in our `.pt` file of learned linear attention map and LoRA weights:
```
sbatch scripts/llama3_1_405b/rp_trenchcoat/405_mmlu_eval.sh
bash scripts/llama3_1_70b/rp_2048_contig/cria_prepare/eval.sh
```

These call to the `lolcats/inference/evals/eval_mmlu.py` file, which just loops through mmlu.pkl and uses the last-token model logits to get the predictions.


### VLLM Integration 

Coming Soon!
