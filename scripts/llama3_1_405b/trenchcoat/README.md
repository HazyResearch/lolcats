


For the LoLCATS-Crias in a Trenchcoat approach, instead of distilling the attention maps all at once, we do it in chunks of C layers at once. Recall that a baby Llama is called a Cria.

The steps to do this are as follows:
1. Attention distill by Cria: We train the linear attention feature maps to match the attention maps from softmax attention. The training loss is the sum of the MSE and XENT losses for the Cria of $C$ layers. At Cria $i$, the training data inputs are the outputs from the last Cria $i-1$.
2. LoRA by Cria: Our script will also run LoRA finetuning for the Cria; again the training data is from Cria $i-1$.
3. LoRA End-to-End: Run end-to-end LoRA finetuning to stich the Cria back into a Llama (``Cria in a Trenchcoat''). Here the training data is the standard langauge modeling data. 


### Some napkin math

If I let $C=9$, where the 405B Llama has a total of $126$ Transformer layers and $16384$ hidden size, and I use $T$ training data tokens in total, then the amount of resources I need are:

- Disk space: $14 \times 16384 \times T \times 2$ bytes of disk space. Roughly $4$ terrabytes if $T = 40M$. 
- GPUs: You will need to load the full base Llama 405B model in order to (1) collect the Cria inputs, (2) shard the model into groups of $9$ layers, and (3) at the very end, to do LoRA finetuning -- we used $3$ nodes of $8 \times 80GB$ GPUs for this. During attention distillation, $9$ layers takes roughly $50$GB of disk space at $16$-bit precision, and *just* fits in GPU during training.
- CPU: We save out the training data (inputs from prior Cria) in torch datasets. When we load this in during training, it consumes CPU RAM -- if you run into issues then use less data or re-engineer the datasets logic.


### Scripts

```
bash 

cd lolcats/

# Save the inputs from each Cria. (3 nodes)
bash scripts/llama3_1_405b/trenchcoat/cria_prepare/collect_inputs.sh

# Shard the Llama 405B into 126/C chunks of C layers. (3 nodes)
bash scripts/llama3_1_405b/trenchcoat/cria_prepare/collect_inputs.sh

# Attention distillation and Cria-level LoRA. (1 80GB GPU per run)
bash scripts/llama3_1_405b/trenchcoat/cria_train/1.sh 
...
bash scripts/llama3_1_405b/trenchcoat/cria_train/14.sh 

# End-to-end LoRA to stitch it all back together. (3 nodes).
bash scripts/llama3_1_405b/trenchcoat/cria_prepare/stitch.sh

```



