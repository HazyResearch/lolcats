"""
Data utils for Llama3
"""

def encode_header(message: str, tokenizer) -> list[int]:
    tokens = []
    tokens.append(tokenizer.get_added_vocab()["<|start_header_id|>"])
    tokens.extend(tokenizer.encode(message["role"], add_special_tokens=False))
    tokens.append(tokenizer.get_added_vocab()["<|end_header_id|>"])
    tokens.extend(tokenizer.encode("\n\n", add_special_tokens=False))
    return tokens


def encode_message(message: str, tokenizer, include_header: bool = True) -> list[int]:
    tokens = encode_header(message, tokenizer) if include_header else []
    tokens.extend(
        tokenizer.encode(message["content"].strip(), add_special_tokens=False)
    )
    tokens.append(tokenizer.get_added_vocab()["<|eot_id|>"])
    return tokens


def template_and_tokenize(sample, tokenizer, include_label: bool = True, 
                          system_prompt: str = None):
    if system_prompt is not None:
        dialog = [{'role': 'system', 'content': system_prompt}]
    else:
        dialog = []

    chat = []
    instruction = sample['instruction']
    if sample['input'] != '':
        instruction += f"\n\n{sample['input']}"
    dialog.extend([
        {'role': 'user', 'content': instruction},
        {'role': 'assistant', 'content': sample['output']},
    ])
        
    prompt = []
    prompt.append(tokenizer.get_added_vocab()["<|begin_of_text|>"])
    for message in dialog[:-1]:
        prompt.extend(encode_message(message, tokenizer))

    if include_label:
        answer = encode_message(dialog[-1], tokenizer)
        answer.append(tokenizer.get_added_vocab()["<|end_of_text|>"])
    else:
        answer = []
        target = encode_message(dialog[-1], tokenizer, include_header=False)
        target.append(tokenizer.get_added_vocab()["<|end_of_text|>"])
        # Add the start of an assistant message for the model to complete.
        prompt.extend(encode_header({"role": "assistant", "content": ""}, tokenizer))

    input_ids = prompt + answer
    attn_mask = [1] * len(input_ids)

    sample =  {
        "input_ids": input_ids,
        "attention_mask" : attn_mask,
        "labels": [-100] * len(prompt) + answer if include_label else target,
    }
    return sample