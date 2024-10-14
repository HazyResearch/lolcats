## Coming Soon! VLLM Integration


#### 1. Clone VLLM
Also run VLLM installations.
```bash
git clone https://github.com/vllm-project/vllm
``` 

#### 2. Copy the following LoLCATS specific files into vllm.

```
bash 
cp lolcats/inference/vllm_files/lolcats.py vllm/model_executor/models/lolcats.py
```

And add the new LoLCATS models from: 
```bash
lolcats/inference/vllm_files/__init__.py -> vllm/model_executor/models/__init__.py
```

####  3. Set the model checkpoint paths. 

Given your local download of the 405B weights, go to the ```Meta-Llama-3.1-405B/config.py``` file and modify the architecture list from ```LlamaForCausalLM``` to ```LlamaLolcatsForCausalLM```. 

In ```vllm/model_executor/models/lolcats_inference_paged.py``` set the ```PATH=....pt``` to the name of your copy of the linearized weights (feature maps and LoRA).

#### 4. Run VLLM. 

These instructions assume you have 2 nodes of $8 \times 80$GB to fit the FP16 405B model. You are okay with 1 node for 70B parameters.
```bash

# Step 1. Follow the VLLM installation quick start to install it in your environment.

# Step 2. Set up a 2 node ray cluster. On the respective nodes, run:
ray start --head                    # on node 1
ray start --address='ip from above' # on node 2

# Step 3. Load the model on the 2 nodes, creating an OpenAI endpoint. Remember to hard code the ckpt paths in lolcats.py PATH (cant use env variable on multinode). Set tensor-parallel-size to 8 if using 1 node. Run this on the head node (node 1).
vllm serve /path/to/hf/model/Meta-Llama-3.1-405B --tensor-parallel-size 16 --enforce-eager # on node 1
```

#### 5. Clone LM-Eval harness and run inference evaluations: 
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
git checkout b281b092
pip install -e .[api]
```

Note that if ```datasets.load_datasets``` gives an issue, it helps to ```pip install -U datasets```.

Launch the evaluation commands on node 1 (the head node of the ray cluster).
```bash
lm_eval --model local-completions --tasks piqa,hellaswag,winogrande,arc_challenge,arc_easy --model_args model='/path/to/hf/model/Meta-Llama-3.1-405B',base_url=http://localhost:8000/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False --batch_size 1 --output save/

lm_eval --model local-completions --tasks mmlu --num_fewshot 5 --model_args model='/path/to/hf/model/Meta-Llama-3.1-405B',base_url=http://localhost:8000/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False --batch_size 1 --output save/ 
```

#### References
Please cite the following if you use their work:

```
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}
```

```
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```
