
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
from datasets import load_dataset
from transformers import RagRetriever, RagSequenceForGeneration, RagTokenizer, AutoTokenizer, pipeline
sys.path.append('src/')  
from DPO.env import OPENAI_API_KEY
from DPO.utils import save_to, process_gpt_output, get_gpt_response
warnings.filterwarnings("ignore")
openai.api_key = OPENAI_API_KEY

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

def evaluate(dataset, model, tokenizer, type_='ppo', model_name='llama', eval_from_file=False, use_rag=False, **kwargs):
    f_rate = 0
    f_rates = {}
    data = []

    if use_rag: 
        rag_dataset = load_dataset("wiki_dpr", 'psgs_w100.multiset.compressed', split='train', cache_dir='cache', trust_remote_code=True)
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq", indexed_dataset=rag_dataset) 
        rag_model = RagSequenceForGeneration.from_pretrained('facebook/rag-token-nq', indexed_dataset=rag_dataset) 
        rag_tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
        summarizer = pipeline('summarization', max_length = 80)

        def get_summary(prompt):
            inputs = rag_tokenizer(prompt, max_length=80, padding=True, return_tensors='pt')
            input_ids = inputs['input_ids']
            question_hidden_states = rag_model.question_encoder(input_ids)[0] 
            docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors='pt')
            doc_scores = torch.bmm(
                    question_hidden_states.unsqueeze(1),
                    docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
            ).squeeze(1)

            doc_ids = docs_dict["doc_ids"]
            summary = summarizer(rag_dataset[doc_ids[0]]['text'], max_length=80)
            summaries = [x['summary_text'] for x in summary]
            return summaries[0]

    if eval_from_file:
        with open(f'results/{model_name}/{type_}_args.json', 'r') as f:
            arguments = json.load(f) 
    
    for i, entry in tqdm(dataset.iterrows(), total=len(dataset)):
        topic = entry.topic
        stance = 'supporting' if entry.label==1 else 'counter'
        context = None
        if use_rag:
            context = get_summary(topic)
            prompt = f"<s> [INST] ### Prompt:  Generate a {stance} argument for the topic: {topic} \n### Context: {context} [/INST]\n### Argument: "
        else:
            prompt = f"<s> [INST] ### Prompt:  Generate a {stance} argument for the topic: {topic} [/INST]\n### Argument: "


        print(prompt)
        exit()
        if eval_from_file:
            y = arguments[i]
        else:
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

    save_to(data, name=f'f-rate.json', output_dir=f'results/{model_name}/arguments/{type_}/')
    print(f_rates)
    print(f"f rate for {type_}:", f_rate)
    print("FALLACY TYPES")
    
    save_to(f_rates, name=f'fallacy_counts.json', output_dir=f'results/{model_name}/arguments/{type_}/')
    for k,v in f_rates.items():
        print(k.upper(), ':', v)


def evaluate_out_of_domain(eval_type, type_='ppo', model_name='llama'):
    assert eval_type == 'debates' or eval_type == 'essays'
    f_rate = 0
    f_rates = {}
    data = []
    
    
    with open(f'results/out_of_domain/llama/{eval_type}/generated_{eval_type}_{type_}.json', 'r') as f:
        data = json.load(f) 
    
    results = []
    for entry in tqdm(data):
        try:
            topic = entry['topic']
            stance = entry['stance']
            y = entry['generated']
            feedback = get_gpt_feedback(topic, y, stance=stance, type_=type_)
            if feedback['fallacy_type']!='None' :
                f_rate+=1
            if feedback['fallacy_type'] in f_rates.keys():
                f_rates[feedback['fallacy_type']] += 1
            else:
                f_rates[feedback['fallacy_type']] = 1

            results.append(feedback)
        except:
            continue

    save_to(results, name=f'f-rate.json', output_dir=f'results/out_of_domain/{model_name}/{eval_type}/{type_}/')
    print(f_rates)
    print(f"f rate for {type_}:", f_rate)
    print("FALLACY TYPES")
    
    save_to(f_rates, name=f'fallacy_counts.json', output_dir=f'results/out_of_domain/{model_name}/{eval_type}/{type_}/')
    for k,v in f_rates.items():
        print(k.upper(), ':', v)

