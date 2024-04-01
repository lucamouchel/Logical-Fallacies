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
        

def generate(prompt: str, model, tokenizer, **generate_kwargs):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=256, truncation=True).to(model.device)
    with torch.no_grad():
        output = model.generate(**tokenized_prompt,
                                **generate_kwargs,
                                pad_token_id=tokenizer.eos_token_id,)
    output_decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_decoded


def get_gpt_feedback(topic, argument, stance, type_='dpo'):
    fallacy_types = pd.read_csv('data/LOGIC/mappings.csv')['Original Name'].unique()
    fallacy_types = [f for f in fallacy_types if f != 'miscellaneous']
    
    s0 = f"Consider the following topic and {stance} argument:\nTopic: {topic}\nArgument: {argument}\n"
    s1 = f"""Out of all the following logical fallacy types\n{fallacy_types}\nwould you qualify this argument as one of these logical fallacies? If not - return "None".\n"""
    s2 = f"If yes, which logical fallacy type is it? Let fallacy_type be your answer.\n"
    s3 = f"""Your task is to complete the following json: {'{'} "type": "{type_}",\n "fallacy_type": <> {'}'}. If an argument does not make sense, set fallacy_type to "None"."""
    prompt = s0 + s1 + s2 + s3
    response = process_gpt_output(get_gpt_response(prompt, model='gpt-4'))
    if response is None:
        print("RETURNED NONE")
        response = {'type': type_, 'fallacy_type': 'None'}
        
    response['topic'] = topic
    response['argument'] = argument
    return response

def evaluate(dataset, model, tokenizer, type_='ppo', model_name='llama', **kwargs):
    f_rate = 0
    f_rates = {}
    data = []
    for i, entry in tqdm(dataset.iterrows()):
        topic = entry.topic
        stance = 'supporting' if entry.label==1 else 'counter'
        prompt = f"<s> [INST] ###Prompt:  Generate a {stance} argument for the topic: {topic} [/INST]\n### Argument: "
        y = generate(prompt, model, tokenizer, **kwargs)
        y = y.split('### Argument: ')[-1].strip()
        feedback = get_gpt_feedback(topic, y, stance=stance, type_=type_)
        if feedback['fallacy_type']!='None' :
            f_rate+=1
        if feedback['fallacy_type'] in f_rates.keys():
            f_rates[feedback['fallacy_type']] += 1
        else:
            f_rates[feedback['fallacy_type']] = 1

        data.append(feedback)

    save_to(data, name=f'{type_}-f-rate.json', output_dir=f'results/{model_name}/arguments/')
    print(f_rates)
    print(f"f rate for {type_}:", f_rate)
    print("FALLACY TYPES")
    
    save_to(f_rates, name=f'{type_}-fallacy_counts.json', output_dir=f'results/{model_name}/arguments/')
    for k,v in f_rates.items():
        print(k.upper(), ':', v)

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