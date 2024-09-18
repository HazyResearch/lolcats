import os
import math
from openai import OpenAI

def calculate_perplexity(logprobs):
    total_log_prob = 0
    token_count = 0

    for token_logprobs in logprobs[1:]:
        if token_logprobs:
            total_log_prob += list(token_logprobs.values())[0].logprob
            token_count += 1

    if token_count == 0:
        return float('inf')

    print(token_count)
    perplexity = math.exp(-total_log_prob / token_count)
    return perplexity

def calc_perplexity_serve(logprobs, trim=1):
    logprobs = logprobs[:-trim]
    logprobs = [x for x in logprobs if x is not None]
    print(f"{len(logprobs)=}")
    return math.exp(-sum(logprobs) / len(logprobs))

if __name__ == '__main__':
    use_served_model = True
    model_size = 70 # [8, 70]
    PATH = f"/data/rahul/models/Meta-Llama-3.1-{model_size}B/"
    CKPT_PATH = f'/data/rahul/checkpoints/{model_size}b.pt'
    openai_api_base = "http://0.0.0.0:8000/v1"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" 
    os.environ["LOLCATS_ADAPTER_PATH"] = CKPT_PATH

    prompts = [
        "I'm Michael:- 3rd-year Computer Science PhD student advised by Chris Ré.- Labmate at HazyResearch, Stanford AI Lab, Stanford Machine Learning Group. I currently work on deep learning architectures for expressive + efficient long sequence modeling, and using these advances to enable learning from new tasks and data types and I also care about deep learning robustness. I received my A.B. in", 
        # Statistics and Computer Science at Harvard in 2020. I'm grateful to have"
        # "The 2024 Summer Paralympics (French: Jeux paralympiques d'été de 2024), also known as the Paris 2024 Paralympic Games, and branded as Paris 2024, is the 17th Summer Paralympic Games, an international multi-sport parasports event governed by the International Paralympic Committee, being held in Paris, France, from 28 August to 8 September 2024. These games mark the first time Paris is hosting the Summer Paralympics and the second time that France is hosting the new ",
        # "Manchester United Football Club, commonly referred to as  Man United (often Man United (often stylised as Man Utd), or simply United, is a Man United (often stylised as Man Utd), or simply United, is a professional football club based in Old Trafford, Greater Manchester, England. They compete in the Premier League, the top tier of English football. Nicknamed the Red Devils, they were founded as Newton Heath LYR Football Club in 1878, but changed their name to Manchester United in 1902. After a spell playing in Clayton, Manchester, the club moved to their current stadium, Old Trafford, in 1910. "
    ]

    if use_served_model:
        client = OpenAI(base_url=openai_api_base, api_key="EMPTY")
        models = client.models.list()
        model = models.data[0].id
        tokens = 3
        outputs = client.completions.create(
            model=model,
            prompt=prompts,
            temperature=0,
            logprobs=1,
            max_tokens=tokens,
            seed=0,
            echo=True,
        )
        for prompt, choice in zip(prompts, outputs.choices):
            logprobs = choice.logprobs.token_logprobs
            print(f"Prompt: {len(prompt.split())}\n{prompt}")
            print(f"Completion: {choice.text.replace(prompt, '')}")
            print(f'Perplexity: {calc_perplexity_serve(logprobs, trim=tokens)}')
            print("\n")
    else:

        os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

        from vllm import ModelRegistry, LLM, SamplingParams

        from src.model.modeling_llama_vllm import LlamaLolcatsForCausalLM
        ModelRegistry.register_model("LlamaLolcatsForCausalLM", LlamaLolcatsForCausalLM)
        sampling_params = SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1, min_tokens=1, max_tokens=1)
        llm = LLM(model=PATH, tensor_parallel_size=8, enforce_eager=True)
        outputs = llm.generate(
            prompts,
            sampling_params,
        )
        logprobs = output.prompt_logprobs
        for output in outputs:
            print(f"Perplexity: {calculate_perplexity(output.prompt_logprobs):.4f}")

    # Print the outputs.
