import sys
sys.path.append("../")

import torch
import gradio as gr
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from src.utils.setup import seed_everything
from src.utils.logging import print_header
from src.model.pretrained import get_pretrained_loader
from src.model.load_model import load_and_convert_attns, load_and_convert_finetune

def load_model_from_checkpoint(
    attn_mlp_checkpoint_path: str = None, 
    finetune_checkpoint_path: str = None, 
    model_config_path: str = None,
    distill_config_path: str = None,
    finetune_config_path: str = None,
    config_dir: str = 'configs',
    print_model: bool = False, 
    debug: bool = False,
    huggingface_token: str = None,
    use_cuda_kernels: bool = False,
    use_attention: bool = False
):

    is_local = attn_mlp_checkpoint_path.endswith(".pt")
    
    model_config = OmegaConf.load(model_config_path)
    distill_config = OmegaConf.load(distill_config_path)
    finetune_config = OmegaConf.load(finetune_config_path)

    model_loader = get_pretrained_loader(**model_config.model, 
                                         huggingface_token=huggingface_token)
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    if use_attention:
        model = model_loader.load('softmax')
        return model, model_config, tokenizer

    model = model_loader.load(model_config['attention']['attention_type'])
    if use_cuda_kernels:
        print('*** Using TK CUDA kernels **')
        model_config['attention']['attention_type'] = 'lolcats_llama_window_tk_gen'

    if is_local:
        checkpoint_path = attn_mlp_checkpoint_path
    else: 
        checkpoint_path = None
    model, distill_peft_config = load_and_convert_attns(
        model, model_config,
        attention_type=None, 
        checkpoint_path=checkpoint_path,
        print_model=debug,
        merge_loras=False,
        peft_gradient_checkpointing=False,
        train_attention=False)
    
    if is_local:
        checkpoint_path = attn_mlp_checkpoint_path
    else: 
        checkpoint_path = None
    model, ft_peft_config = load_and_convert_finetune(
        model, finetune_config,
        checkpoint_path=checkpoint_path,
        print_model=debug,
        merge_loras=False,
        peft_gradient_checkpointing=False)

    if not is_local:
        model = load_hf_weights(
            model, 
            attn_mlp_checkpoint_path, finetune_checkpoint_path, 
            filename="model.pt"
        )
    if use_cuda_kernels:
        print('*** Using TK CUDA kernels ***')

    if print_model:
        print('*** Model after checkpoint load ***')
        print(model)

    return model, model_config, tokenizer

def load_hf_weights(model, distill_repo_id, ft_repo_id, filename="model.pt"):
    for repo_id in [distill_repo_id, ft_repo_id]:
        if repo_id is None: continue 

        print(f"Loading weights from {repo_id}")

        local_file_path = hf_hub_download(repo_id=repo_id, filename=filename)    
        state_dict = torch.load(local_file_path)
        if 'model_state_dict' in state_dict: 
            state_dict = state_dict['model_state_dict']
        else:
            pass
        _keys = model.load_state_dict(state_dict, strict=False)
        if len(_keys.unexpected_keys) > 0:
            new_state_dict = {k.replace('model.', 'model.model.'): v for k, v in state_dict.items()}
            _keys = model.load_state_dict(new_state_dict, strict=False)
        if len(_keys.unexpected_keys) > 0:
            new_state_dict = {k.replace('model.', 'base_model.model.model.'): v for k, v in state_dict.items()}
            _keys = model.load_state_dict(new_state_dict, strict=False)

        try:
            assert len(_keys.unexpected_keys) == 0
            print('*** All expected keys matched successfully ***')
        except Exception as e:
            print(e)
            print('*** Error: unexpected keys in checkpoint - please fix ***')
            print('Unexpected keys:')
            for k in _keys.unexpected_keys:
                print(k)
            exit()

    return model

def load_model_and_tokenizer():
    CONFIG_DIR = 'configs'  # Update to your path

    model_config_path = f"{CONFIG_DIR}/model/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01.yaml"
    distill_config_path = f"{CONFIG_DIR}/experiment/distill_alpaca_clean_xent0_mse1000_lr1e-2.yaml"
    finetune_config_path = f"{CONFIG_DIR}/experiment/finetune_lora_qkvo_alpaca_clean.yaml"
    attn_mlp_checkpoint_path = 'hazyresearch/lolcats-llama-3.1-8b-distill'
    finetune_checkpoint_path = 'hazyresearch/lolcats-llama-3.1-8b-ft-lora'

    model, model_config, tokenizer = load_model_from_checkpoint(
        attn_mlp_checkpoint_path=attn_mlp_checkpoint_path, 
        finetune_checkpoint_path=finetune_checkpoint_path, 
        model_config_path=model_config_path,
        distill_config_path=distill_config_path,
        finetune_config_path=finetune_config_path,
        config_dir=CONFIG_DIR,
        print_model=False, 
        debug=False,
        huggingface_token=None,
        use_cuda_kernels=False,
        use_attention=False
    )
    model = model.to('cuda')
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def generate_response(prompt):
    all_prompts = [prompt]

    with torch.no_grad():
        model_input = tokenizer(all_prompts, return_tensors="pt").to(model.device)
        model_output = model.generate(
            **model_input, use_cache=True, 
            max_new_tokens=50,
            do_sample=False,
            top_k=1,
            top_p=1.0,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id)
    generated_tokens = model_output[0]
    input_len = model_input['input_ids'].shape[1]
    generated_tokens = generated_tokens[input_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text

iface = gr.Interface(fn=generate_response, inputs="text", outputs="text", title="LOLcats Model Demo")

iface.launch()
