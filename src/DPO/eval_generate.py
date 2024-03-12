import torch
import transformers
import pandas as pd
import numpy as np
import os
from collections import namedtuple
import torch
from tqdm import tqdm
import warnings
import json
import openai
import time
import os
import transformers
from env import OPENAI_API_KEY
import pathlib
import argparse
from peft import AutoPeftModelForCausalLM
def generate(prompt: str, model, tokenizer):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=80, truncation=True).to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids=tokenized_prompt.input_ids,
                                attention_mask=tokenized_prompt.attention_mask,
                                max_new_tokens=30,
                                #pad_token_id=tokenizer.eos_token_id,
                                #top_p=0.5,
                                #temperature=1,
                                no_repeat_ngram_size=2,
                                do_sample=False)

    output_decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_decoded

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft-path', default=None)
    parser.add_argument('--dpo-path', default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    ref_model_dir = args.sft_path
    dpo_model_dir = args.dpo_path
    
    if ref_model_dir:
        model = AutoPeftModelForCausalLM.from_pretrained(ref_model_dir, device_map='auto')
        tokenizer = transformers.AutoTokenizer.from_pretrained(ref_model_dir)
        
    if dpo_model_dir:
        model = AutoPeftModelForCausalLM.from_pretrained(dpo_model_dir, device_map='auto')
        tokenizer = transformers.AutoTokenizer.from_pretrained(ref_model_dir)

    test_set = pd.read_json('data/argumentation/test_cckg.json')
    test_set = test_set.reset_index()
    
    for i, entry in tqdm(test_set.iterrows()):
            topic = entry.topic
            stance = 'SUPPORTING' if entry.label == 1 else 'COUNTER'
            prompt = f"<s> [INST] ###Prompt:  Generate a {stance} argument for the topic: {topic} [INST]\n### Argument: "
            generated = generate(prompt, model, tokenizer)

            print(generated)

if __name__ == "__main__":
    main()
