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
import pathlib
import argparse
from DPO.utils import save_to
from peft import AutoPeftModelForCausalLM


def generate(prompt: str, model, tokenizer):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=256, truncation=True).to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids=tokenized_prompt.input_ids,
                                attention_mask=tokenized_prompt.attention_mask,
                                max_new_tokens=30,
                                pad_token_id=tokenizer.eos_token_id,
                                #top_p=0.5,
                                #temperature=1,
                                no_repeat_ngram_size=2,
                                do_sample=True)

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
    
    debate_data = pd.read_csv('data/test_debate_concept.txt', header=None, sep='\t')
    
    ref_model = AutoPeftModelForCausalLM.from_pretrained(ref_model_dir, device_map='auto')        
    model = AutoPeftModelForCausalLM.from_pretrained(dpo_model_dir, device_map='auto')
    tokenizer = transformers.AutoTokenizer.from_pretrained(ref_model_dir)
    
    

    sft_generated = []
    dpo_generated = []
    
    seen_topics = set()
    for i, entry in tqdm(debate_data.iterrows()):
        topic = entry[3]
        if topic in seen_topics:
            continue
        if len(topic) > 200:
            break
        stance = 'SUPPORTING' if entry[6] == 'support' else 'COUNTER'
        prompt = f"<s> [INST] ###Prompt: Generate a {stance} argument for the topic: {topic} [INST]\n### Argument: "
        generated_dpo = generate(prompt, model, tokenizer)
        generated_sft = generate(prompt, ref_model, tokenizer)

        sft_generated.append({'topic': topic, 'stance': stance, 'generated': generated_sft.split('### Argument:')[-1].strip()})
        dpo_generated.append({'topic': topic, 'stance': stance, 'generated': generated_dpo.split('### Argument:')[-1].strip()})

        print(generated_dpo.split('### Argument:')[-1].strip())

if __name__ == "__main__":
    main()
