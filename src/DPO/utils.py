import torch
import os
import json 
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments
import nltk
import re
import pathlib
import openai 
import time
from tqdm import tqdm

def save_to(data, name, output_dir):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, name)
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, sort_keys=False)
    
def chat_completion(messages, model="gpt-3.5-turbo", return_text=True, model_args=None):
    if model_args is None:
        model_args = {}
    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages, **model_args)
            if return_text:
                return response["choices"][0]["message"]["content"].strip()
            return response
        except Exception as e:
            print(e)
            print("Timed out. Waiting for 1 minute.")
            time.sleep(60)
            continue

def get_gpt_response(input_, model='gpt-3.5-turbo', necessary_tokens=None):
    return chat_completion([{"role": "assistant", "content": input_}], model=model, return_text=True, model_args={
                    "temperature": 0.4,
                    "max_tokens": 150 if not necessary_tokens else necessary_tokens,
                    "top_p": 0.4,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                    })
        
        
def process_gpt_output(output_text):
    try:
        result_dict = json.loads(output_text)
        return result_dict
    except json.JSONDecodeError:
        start_index = output_text.find('{')
        end_index = output_text.rfind('}')
        if start_index != -1 and end_index != -1:
            extracted_dict_text = output_text[start_index:end_index + 1]
            try:
                result_dict = json.loads(extracted_dict_text)
                return result_dict
            except json.JSONDecodeError:
                print("Failed to extract a valid dictionary from the text.")
                return None
        else:
            print("No dictionary found in the text.")
            return None
        
def remove_incomplete_last_sentence(text):
    sentences = nltk.sent_tokenize(text)

    last_sentence = sentences[-1]
    if last_sentence.endswith(('?', '.', '!')):
        return text
    else:
        return ' '.join(sentences[:-1])

def sanitize(sentence):
    sanitized_sentence = re.sub(r'[^\w\s.,!?\'-]', '', sentence)
    sanitized_sentence = re.sub(r'\s+', ' ', sanitized_sentence).strip()
    return sanitized_sentence

def get_data(data_dir, split='train', return_type='dataset', with_equivalence=False):
    if data_dir[-1] != '/': 
        data_dir += '/'
    with open(f'{data_dir+split}.json', 'r') as f:
        df = json.load(f)

    df = pd.DataFrame(df)
    if with_equivalence: 
        with open(data_dir + f'{split}_equivalent.json', 'r') as f:
            equivalence = json.load(f)
        
        df2 = df.copy()    
        df2.drop_duplicates(subset=['chosen'], inplace=True)
        
        d = {'prompt': df2.prompt, 'chosen': df2.chosen, 'rejected': [equivalent['argument'] for equivalent in equivalence]}
        
        new_df = pd.DataFrame(data=d).reset_index(drop=True)
        df = pd.concat([df, new_df])
 
        df = df.reset_index(drop=True)

    if return_type == 'df':
        return df
    
    for i, item in df.iterrows():
        prompt = df.iloc[i]['prompt']
        if 'supporting argument' in prompt:
            prompt = prompt.replace('supporting argument', 'SUPPORTING argument of about 20 words')
        else:
            prompt = prompt.replace('counter argument', 'COUNTER argument of about 20 words')

        prompt += '\n### Argument:'
        df.iloc[i]['prompt'] = prompt
    
    return Dataset.from_dict(df)

def get_training_args(args):
    return TrainingArguments(
            output_dir=args.output_dir,               
            overwrite_output_dir=False,                  
            num_train_epochs=args.n_epochs,                   
            per_device_train_batch_size=args.batch_size,         
            learning_rate=args.learning_rate,                      
            warmup_steps=args.warmup_steps,                           
            weight_decay=args.weight_decay,                         
            adam_epsilon=args.adam_epsilon,                         
            save_steps=args.save_steps,                       
            logging_steps=args.logging_steps,                      
            save_total_limit=2,                         
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )       