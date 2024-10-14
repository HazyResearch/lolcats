"""
Quick demo of linearized LLM generations
"""
from typing import Optional, List
from os.path import join
import time
import argparse
import torch

from omegaconf import OmegaConf

from transformers import TextStreamer, TextIteratorStreamer, AutoTokenizer

from src.utils.setup import seed_everything
from src.utils.logging import print_header
from src.model.pretrained import get_pretrained_loader
from src.model.load_model import load_and_convert_attns, load_and_convert_finetune


system_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""


def get_args():
    parser = argparse.ArgumentParser()
    # Model load + setup
    parser.add_argument("--attn_mlp_checkpoint_path", type=str, default=None)
    parser.add_argument("--finetune_checkpoint_path", type=str, default=None)
    parser.add_argument("--config_dir", type=str, default='configs')
    parser.add_argument("--seed", type=int, default=42)

    # Generation
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    # Miscellaneous
    parser.add_argument("--benchmark", action='store_true', default=False)
    parser.add_argument("--print_model", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--huggingface_token", type=str, default=None)

    # Alt
    parser.add_argument("--attn_checkpoint_path", type=str, default=None)
    parser.add_argument("--peft_checkpoint_path", type=str, default=None)

    args = parser.parse_args()
    if args.attn_mlp_checkpoint_path is None and args.attn_checkpoint_path is not None:
        args.attn_mlp_checkpoint_path = args.attn_checkpoint_path
    if args.finetune_checkpoint_path is None and args.peft_checkpoint_path is not None:
        args.finetune_checkpoint_path = args.peft_checkpoint_path
    return args


def get_lm_eval_lolcats_model(model_kwargs: dict, lolcats_model: bool = True):
    lm_kwargs = copy.deepcopy(model_kwargs)
    lm_kwargs['pretrained'] = lm_kwargs['pretrained_model_name_or_path']
    lm_kwargs['dtype'] = str(lm_kwargs['torch_dtype']).split('.')[-1]
    del lm_kwargs['torch_dtype']

    if 'Llama' in lm_kwargs['pretrained_model_name_or_path']:  #  and lolcats_model:
        lm_kwargs['device_map'] = None
        from lm_eval_harness.models import ShardedLolcatsLlamaForCausalLM
        lm = ShardedLolcatsLlamaForCausalLM.create_from_arg_string(
            '', lm_kwargs,
        )
    else:
        sys.path.append(LM_EVALUATION_HARNESS_PATH)
        from lm_eval.models import get_model
        
        lm = get_model('hf-causal-experimental').create_from_arg_string(
            '', lm_kwargs,
        )
    # model = lm.model
    return lm


class BatchTextIteratorStreamer(TextIteratorStreamer):
    """
    Copied from https://discuss.huggingface.co/t/textiteratorstreamer-compatibility-with-batch-processing/46763/2
    """
    def __init__(self, 
                 tokenizer: AutoTokenizer, 
                 batch_size: int, 
                 skip_prompt: bool = False, 
                 timeout: Optional[float] = None, 
                 **decode_kwargs: any):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)
        self.batch_size = batch_size
        self.token_cache = [[] for _ in range(batch_size)]
        self.print_len = [0 for _ in range(batch_size)]
        self.generate_exception = None
        self.go_up = 0 + batch_size
        self.stop_signal = tokenizer.eos_token

    def put(self, value):
        if len(value.shape) != 2:
            value = torch.reshape(value, (self.batch_size, value.shape[0] // self.batch_size))

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        printable_texts = list()
        for idx in range(self.batch_size):
            self.token_cache[idx].extend(value[idx].tolist())
            text = self.tokenizer.decode(self.token_cache[idx], **self.decode_kwargs)

            if text.endswith("\n"):
                printable_text = text[self.print_len[idx] :]
                self.token_cache[idx] = []
                self.print_len[idx] = 0
                self.go_up += 1
                # If the last token is a CJK character, we print the characters.
            elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
                printable_text = text[self.print_len[idx] :]
                self.print_len[idx] += len(printable_text)
            else:
                printable_text = text[self.print_len[idx] : text.rfind(" ") + 1]
                # printable_text = text[self.print_len[idx] : self.print_len[idx] + 1]
                # if printable_text == '':
                    # printable_text = self.stop_signal
                self.print_len[idx] += len(printable_text)
            printable_texts.append(printable_text)

        self.on_finalized_text(printable_texts)

    def end(self):
        printable_texts = list()
        for idx in range(self.batch_size):
            if len(self.token_cache[idx]) > 0:
                text = self.tokenizer.decode(self.token_cache[idx], **self.decode_kwargs)
                printable_text = text[self.print_len[idx] :]
                self.token_cache[idx] = []
                self.print_len[idx] = 0
            else:
                printable_text = ""
                # printable_text = self.stop_signal
            printable_texts.append(printable_text)

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_texts, stream_end=True)

    def on_finalized_text(self, texts: List[str], stream_end: bool = False):
        self.text_queue.put(texts, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

        try:
            text = [
                ''.join([x[i] if i < len(x) else self.stop_signal 
                         for x in self.text_queue.queue ]) 
                for i in range(len(self.text_queue.queue[0]))
            ]
            # text = '\n\n'.join(self.text_queue.queue[0])
            text = '\n------------\n'.join(text)
            go_up = "\033[F" * self.go_up  # len(text)  # Goes up this many lines
            # go_down = "\n" * self.go_up  #  len(text)  # Goes up this many lines
            print(f'{text}', flush=True, end="" if not stream_end else None)
            # print(f'{go_up}{text}', end="" if not stream_end else None)
        except Exception as e:
            print(self.stop_signal)

def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def setup_fsdp_config(config, args, checkpoint_name: str = 'finetune'):
    """
    Hacky arguments for llama-recipes training function
    """
    config.seed = args.seed
    config.enable_fsdp = args.enable_fsdp
    config.low_cpu_fsdp = args.low_cpu_fsdp
    config.dist_checkpoint_root_folder = args.checkpoint_dir
    config.dist_checkpoint_folder = checkpoint_name

    config.model_name = args.run_name
    config.use_peft = False  # We have custom logic for saving PEFT modules
    config.save_model = True
    config.run_validation = True
    config.use_fp16 = False
    config.save_model = True
    config.save_optimizer = False
    config.output_dir = args.checkpoint_dir
    config.save_metrics = not args.no_wandb
    config.gradient_clipping = False
    config.gradient_clipping_threshold = 1.0
    config.num_epochs = getattr(config.trainer, 'num_train_epochs', None)
    config.num_train_steps = getattr(args, 'num_train_steps', None)  # exit training loop early for debugging
    config.eval_steps = getattr(config.trainer, 'eval_steps', None)  # how many gradient updates before evaluating
    return config


def load_model_from_checkpoint(attn_mlp_checkpoint_path: str, 
                               finetune_checkpoint_path: str, 
                               config_dir: str = 'configs',
                               print_model: bool = False, 
                               debug: bool = False,
                               huggingface_token: str = None):
    rank = 0
    # Get configs from checkpoint paths
    try:
        model_config = attn_mlp_checkpoint_path.split('-m=')[-1].split('-f=')[0]
        distill_config = attn_mlp_checkpoint_path.split('-d=')[-1].split('-m=')[0]
    except Exception as e:
        model_config = finetune_checkpoint_path.split('-m=')[-1].split('-f=')[0]
        distill_config = None
    
    model_config = join(config_dir, 'model', f'{model_config}.yaml')
    model_config = OmegaConf.load(model_config)
    
    if distill_config is not None:
        distill_config = join(config_dir, 'experiment', f'{distill_config}.yaml')
        distill_config = OmegaConf.load(distill_config)
    else:
        distill_config = {}

    finetune_config = finetune_checkpoint_path.split('-f=')[-1].split('-')[0]
    finetune_config = join(config_dir, 'experiment', f'{finetune_config}.yaml')
    finetune_config = OmegaConf.load(finetune_config)

    # Load initial model
    model_loader = get_pretrained_loader(**model_config.model, 
                                         huggingface_token=huggingface_token)
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    model = model_loader.load(model_config['attention']['attention_type'])
    if debug:
        print_header('Pretrained Model')
        print(model)

    # Add subquadratic attentions
    model, distill_peft_config = load_and_convert_attns(model, model_config,
                                                        attention_type=None,  # in model_config
                                                        checkpoint_path=attn_mlp_checkpoint_path,
                                                        print_model=debug,
                                                        merge_loras=False,
                                                        peft_gradient_checkpointing=False,
                                                        train_attention=False)
    
    # Add PEFT parameters
    model, ft_peft_config = load_and_convert_finetune(model, finetune_config,
                                                      checkpoint_path=finetune_checkpoint_path,
                                                      print_model=debug,
                                                      merge_loras=False,
                                                      peft_gradient_checkpointing=False)
    if print_model:
        print_header('*** Model after checkpoint load ***')
        print(model)

    return model, model_config, tokenizer


def get_model_name(attn_mlp_checkpoint_path: str, finetune_checkpoint_path: str, 
                   model_config: str = None):
    model_name = '😺 ' if attn_mlp_checkpoint_path is not None else ''
    if 'llama3_8b_' in finetune_checkpoint_path:
        model_name += f'Llama-3-8B'
    elif 'llama3_1_8b_' in finetune_checkpoint_path:
        model_name += f'Llama-3.1-8B'
    elif 'llama2_7b_' in finetune_checkpoint_path:
        model_name += f'Llama-2-7B'
    elif 'mistral_7b_' in finetune_checkpoint_path:
        model_name += f'Mistral-7B'

    if attn_mlp_checkpoint_path is not None:
        model_name += f'-LoLCATs'

    if 'alpaca_clean' in finetune_checkpoint_path:
        model_name += f'-Alpaca'

    elif model_config is not None:
        if 'llama3_8b_' in model_config:
            model_name += f'Llama-3-8B'
        elif 'llama2_7b_' in model_config:
            model_name += f'Llama-2-7B'
        elif 'mistral_7b_' in model_config:
            model_name += f'Mistral-7B'

    return model_name


def main():
    args = get_args()
    seed_everything(args.seed)
    model, model_config, tokenizer = load_model_from_checkpoint(
        args.attn_mlp_checkpoint_path, args.finetune_checkpoint_path, 
        config_dir=args.config_dir, print_model = args.print_model, debug = args.debug,
    )
    model.eval()
    input_len = len(tokenizer(system_prompt)['input_ids'])

    model_name = get_model_name(args.attn_mlp_checkpoint_path, 
                                args.finetune_checkpoint_path, 
                                model_config)
    while True:
        print(f'\n>> Generating {args.num_generations} responses in parallel')
        prompt = input(f'>> Message {model_name} (or cmd-c to quit)... ')
        all_prompts = [system_prompt.format(prompt=prompt)] * args.num_generations

        
        if args.num_generations == 1:
            streamer = TextStreamer(tokenizer, skip_prompt=True,
                                    decode_kwargs={'skip_special_tokens': True})
        else:
            streamer = BatchTextIteratorStreamer(tokenizer=tokenizer, 
                                                 batch_size=args.num_generations,
                                                 skip_prompt=True,)
    
        with torch.no_grad():
            model_input = tokenizer(all_prompts, return_tensors="pt").to(model.device)

            if args.benchmark:
                torch.cuda.synchronize()
                start_time = time.time()
            model_output = model.generate(**model_input, use_cache=True, 
                                          max_new_tokens=args.max_new_tokens, 
                                          do_sample=True,
                                          top_k=args.top_k,
                                          top_p=args.top_p,
                                          num_return_sequences=1,
                                          pad_token_id=tokenizer.eos_token_id,
                                          streamer=streamer)
            if args.benchmark:
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                total_tokens = (model_output != tokenizer.eos_token_id).sum().item()
                print_header('(Coarse) stats for nerds')
                print(f'├── Model data type:                      {model.dtype}')
                print(f'├── Time of longest response:             {elapsed:.3f} sec')
                print(f'├── Total tokens processed + generated:   {total_tokens}')
                print(f'├── Throughput (lagged by last response): {total_tokens / elapsed:.3f} tokens/sec')
                

if __name__ == '__main__':
    main()