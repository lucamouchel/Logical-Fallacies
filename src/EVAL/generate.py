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
import sys
import pathlib 
import argparse
from peft import AutoPeftModelForCausalLM
from datetime import datetime
from transformers import AutoModelForCausalLM

sys.path.append('src/')  
from DPO.utils import save_to
warnings.filterwarnings("ignore")
from EVAL.utils import get_gpt_feedback
from DPO.env import OPENAI_API_KEY 
openai.api_key = OPENAI_API_KEY
GENERATION_KWARGS = {'max_new_tokens': 50, 'no_repeat_ngram_size': 2, 'do_sample': True, 'top_p': 0.75}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=True, help='sft, dpo, cpo, kto,..')
    
    model = AutoPeftModelForCausalLM.from_pretrained('models_BIS/cpo_mistral_2024-06-04 20:51:23.983451', device_map='auto')
    model.eval()
    ### SFT and DPO have the same tokenizer -- but not sure for the other two
    tokenizer = transformers.AutoTokenizer.from_pretrained('models_BIS/cpo_mistral_2024-06-04 20:51:23.983451')
    test_set = pd.read_json('data/argumentation/test_cckg.json')[:200]
    
    def generate(prompt: str, model, tokenizer,n=5, **generate_kwargs):
        """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
        tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=200, truncation=True).to(model.device)
        with torch.no_grad():
            output = model.generate(**tokenized_prompt,
                                    **generate_kwargs,
                                    pad_token_id=tokenizer.eos_token_id,
                                    num_return_sequences=n)
            output_decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        return output_decoded
    
    arguments = []
    for i, entry in tqdm(test_set.iterrows(), total=len(test_set)):
        topic = entry.topic
        stance = 'SUPPORTING' if entry.label == 1 else 'COUNTER'
        prompt = f"<s> [INST] ### Prompt:  Generate a {stance} argument for the topic: {topic} [/INST]\n### Argument: "
        ys = generate(prompt, model, tokenizer, n=1, **GENERATION_KWARGS)
        print("TOPIC:", topic)
    
        y = list(map(lambda y : y.split('### Argument:')[-1].strip(), ys))[0]
        print(y)
        if '#' in y:
            y = y.split('#')[0]
        arguments.append(y)
        print(y)
      
    save_to(arguments, name='mistral_again.json' ,output_dir='results/llama_bis/')
            
if __name__ == "__main__":
    main()