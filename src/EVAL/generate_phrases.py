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

sys.path.append('src/')  
from DPO.utils import save_to
warnings.filterwarnings("ignore")
from EVAL.utils import get_gpt_feedback
from DPO.env import OPENAI_API_KEY 
openai.api_key = OPENAI_API_KEY
GENERATION_KWARGS = {'max_new_tokens': 30, 'no_repeat_ngram_size': 2, 'do_sample': True, 'top_p': 0.8}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--type', required=True, help='sft, dpo, cpo, kto,..')
    args = parser.parse_args()
    
    model = AutoPeftModelForCausalLM.from_pretrained(args.model_path, device_map='auto')
    clf = transformers.AutoModelForSequenceClassification.from_pretrained('models/fallacy_clf/howey_electra-large-mnli', num_labels=2)
    clf_tokenizer = transformers.AutoTokenizer.from_pretrained('models/fallacy_clf/howey_electra-large-mnli')
    model.eval()
    clf.eval()
    ### SFT and DPO have the same tokenizer -- but not sure for the other two
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    test_set = pd.read_json('data/argumentation/test_cckg.json')
    def fallacy_probas(arguments):
        inputs = clf_tokenizer(arguments, add_special_tokens=True, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            out = clf(**inputs)    
            logits = out.logits
            fallacy_logits = logits[:, 1]
        return torch.sigmoid(fallacy_logits).tolist()
    
    def generate(prompt: str, model, tokenizer,n=5, **generate_kwargs):
        """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
        tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=80, truncation=True).to(model.device)
        with torch.no_grad():
            output = model.generate(**tokenized_prompt,
                                    **generate_kwargs,
                                    pad_token_id=tokenizer.eos_token_id,
                                    num_return_sequences=n)
        output_decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        return output_decoded
    
    arguments = []
    data = []
    f_rates = {}
    f_rate = 0
    for i, entry in tqdm(test_set.iterrows(), total=len(test_set)):
        topic = entry.topic
        stance = 'SUPPORTING' if entry.label == 1 else 'COUNTER'
        prompt = f"<s> [INST] ### Prompt:  Generate a {stance} argument for the topic: {topic} [/INST]\n### Argument: "
        with torch.no_grad():
            ys = generate(prompt, model, tokenizer, n=4, **GENERATION_KWARGS)
            
        ys = [y.split('### Argument: ')[-1].strip() for y in ys]
        probas = fallacy_probas(ys)
        y_final = sorted(zip(probas, ys), key=lambda x: x[0])[0] ## select the one with the lowest fallacy probability
        arguments.append(y_final)
        
        feedback = get_gpt_feedback(topic, y_final, stance=stance, type_=args.type)
        if feedback['fallacy_type']!='None' :
            f_rate+=1
        if feedback['fallacy_type'] in f_rates.keys():
            f_rates[feedback['fallacy_type']] += 1
        else:
            f_rates[feedback['fallacy_type']] = 1

        if i % 30 == 0:
            print(f_rates)
        data.append(feedback)
        date = datetime.now()
        
    save_to(data, name=f'f-rate_{date}.json', output_dir=f'results/{args.model_name}/arguments/{args.type}/')
    print(f_rates)
    print(f"f rate for {args.type}:", f_rate)
    print("FALLACY TYPES")
    
    save_to(f_rates, name=f'fallacy_counts_{date}.json', output_dir=f'results/{args.model_name}/arguments/{args.type}/')
    for k,v in f_rates.items():
        print(k.upper(), ':', v)
    if i % 100 == 0:
        print("ARGUMENT:\n", y_final, '\n')    
    save_to(arguments, name=f'{args.type}_args_with_4_samples.json', output_dir=f'results/{args.model_name}/')

        
if __name__ == "__main__":
    main()