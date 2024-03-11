import random
import pandas as pd 
from matplotlib.pyplot import cla
from regex import X
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import os
from tqdm import tqdm
import warnings
from data_loader import load_claim_data, load_claim_data_sampled, load_data
import json
from openai.error import RateLimitError, ServiceUnavailableError, APIError, APIConnectionError, Timeout, InvalidRequestError
import openai
import time
import pathlib
from DPO.env import OPENAI_API_KEY

warnings.filterwarnings("ignore")

####Generate 100 arguments with chatgpt -- generate 100 claims with chatgpt

def generate_prompt(context, stance, context_type):
    """
    context: 
        topic (str) if you want to generate an argument
        evidence (list) if you want to generate a claim
    stance:
        'support' or 'refuting'
    context_type:
        'argument' or 'claim'
    """
    if context_type == 'claim':
        evidence = context
        if len(evidence) > 3: evidence = evidence[:3]    
        text = f"""Given the following evidence, generate a {stance} claim. Evidence: {evidence}
        Your claim should not exceed 20 words. Let gpt_claim be your claim. Complete the following json:
        {"{"}
            "evidence": "{evidence}",
            "claim": gpt_claim
        {"}"}
        """
    elif context_type == 'argument':
        topic = context
        text = f"""Given the following topic, generate a {stance} argument. Topic: {topic}
        Your argument should not exceed 20 words. Let gpt_argument be your argument. Complete the following json:
        {"{"}
            "topic": "{topic}",
            "argument": gpt_argument
        {"}"}
        """
    else: 
        raise ValueError("context_type must be either 'argument' or 'claim'")
    return text

def chat_completion(messages, model="gpt-3.5-turbo", return_text=True, model_args=None):
    if model_args is None:
        model_args = {}
    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages, **model_args)            
            if return_text:
                return response["choices"][0]["message"]["content"].strip()
            return response
        except (RateLimitError, ServiceUnavailableError, APIError, Timeout, InvalidRequestError, APIConnectionError) as e:
            print("Timed out. Waiting for 1 minute.")
            time.sleep(60)
            continue
        
def get_gpt_response(input_, model='gpt-3.5-turbo'):
    return chat_completion([{"role": "assistant", "content": input_}], model=model, return_text=True, model_args={
                "temperature": 0.2,
                "max_tokens": 512,
                "top_p": 0.3,
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
                print(output_text)
                print("Failed to extract a valid dictionary from the text.")
                return None
        else:
            print(output_text)
            print("No dictionary found in the text.")
            return None
        

def save_to(data, name):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, name)
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, sort_keys=False)


def generate_claims_with_GPT(data):
    gpt_result = []
    for _, sample in tqdm(data.iterrows()):
        evidence = sample.evidence
        claim_type = 'support' if  sample.label == 'SUPPORTS' else 'refuting'
        prompt = generate_prompt(context=evidence, stance=claim_type, context_type='claim')
        try:
            gpt_response = get_gpt_response(prompt)
            gpt_response = process_gpt_output(gpt_response)
            if gpt_response is None:
                continue
            gpt_result.append(gpt_response)
        except:
            save_to(gpt_result, 'generated_claims.json')
            continue
        
    print("SAVING to ", os.path.join(output_dir, f'generated_claims.json'))
    save_to(gpt_result, f'generated_claims.json')
    
def generate_arguments_with_GPT(data):
    gpt_result = []
    for _, sample in tqdm(data.iterrows()):
        topic = sample.topic
        arg_type = 'supporting' if sample.label == 1 else 'counter'
        prompt = generate_prompt(context=topic, stance=arg_type, context_type='argument')
        try:
            gpt_response = get_gpt_response(prompt)
            gpt_response = process_gpt_output(gpt_response)
            if gpt_response is None:
                continue
            gpt_result.append(gpt_response)
        except:
            save_to(gpt_result, f'generated_arguments.json')
            continue
    
    print("SAVING to ", os.path.join(output_dir, f'generated_arguments.json'))
    save_to(gpt_result, f'generated_arguments.json')
    
def evaluate_generated_text_with_gpt(sample, context='argument'):
    fallacy_types = pd.read_csv('data/LOGIC/mappings.csv')['Original Name'].unique()
    fallacy_types = [f for f in fallacy_types if f != 'miscellaneous']

    if context == 'argument':
        topic = sample['topic']
        argument = sample['argument']
        s0 = f"Consider the following argument: {argument} given the topic: {topic}\n"
        s1 = f"""Out of all the following logical fallacy types\n{fallacy_types}\nWould you qualify this argument as one of these logical fallacies? If not - return "None".\n"""
        s2 = f"If yes, which logical fallacy type is it? Let fallacy_type be your answer.\n"
        s3 = f"""Your task is to complete the following json: {'{'} "argument": "{argument}",\n "fallacy_type": <> {'}'}. If an argument does not make sense, set fallacy_type to "None"."""

        prompt = s0 + s1 + s2 + s3
        response = get_gpt_response(prompt, model='gpt-4')    
        response = process_gpt_output(response)
        if response is None:
            response = {'argument': argument, 'fallacy_type': 'None'}
            
        response['topic'] = topic        
        
    elif context == 'claim':
        evidence = sample['evidence']
        claim = sample['claim']
        s0 = f"Consider the following claim: {claim} given the evidence {evidence}\n"
        s1 = f"""Out of all the following logical fallacy types\n{fallacy_types}\nWould you qualify this claim as one of these logical fallacies? If not - return "None".\n"""
        s2 = f"If yes, which logical fallacy type is it? Let fallacy_type be your answer.\n"
        s3 = f"""Your task is to complete the following json: {'{'} "claim": "{claim}",\n "fallacy_type": <> {'}'}. If a claim does not make sense, set fallacy_type to "None"."""
        
        prompt = s0 + s1 + s2 + s3
        response = get_gpt_response(prompt, model='gpt-4')
        response = process_gpt_output(response)
        if response is None:
            response = {'claim': claim, 'fallacy_type': 'None'}
        response['evidence'] = evidence
        
    else: 
        raise ValueError("context must be either 'argument' or 'claim'")
    return response

def evaluate_generated_arguments():
    with open(os.path.join(output_dir, 'generated_arguments.json')) as f:
        generated_arguments = json.load(f)
        
    gpt_4_results = []
    for sample in tqdm(generated_arguments):
        response = evaluate_generated_text_with_gpt(sample, context='argument')
        gpt_4_results.append(response)
        
    save_to(gpt_4_results, 'argument_results.json')    
    
def evaluate_generated_claims():
    with open(os.path.join(output_dir, 'generated_claims.json')) as f:
        generated_claims = json.load(f)
        
    gpt_4_results = []
    for sample in tqdm(generated_claims):
        response = evaluate_generated_text_with_gpt(sample, context='claim')
        gpt_4_results.append(response)
        
    save_to(gpt_4_results, 'claim_results.json')
        
        
if __name__ == '__main__':        
    output_dir = 'results/chatgpt_text_generation_evaluation/'
    GENERATE = False
    EVALUATE_FALLACY = False
    openai.api_key = OPENAI_API_KEY
    claim_train_data = load_claim_data_sampled('train', n=100, random_state=42)
    argument_train_data, _, _ = load_data('cckg')
    argument_train_data = argument_train_data.sample(n=100, random_state=42)

    if GENERATE:
        generate_arguments_with_GPT(argument_train_data)
        generate_claims_with_GPT(claim_train_data)
        
    elif EVALUATE_FALLACY:
        evaluate_generated_arguments()
        evaluate_generated_claims()
    else:
        with open(os.path.join(output_dir, 'argument_results.json')) as f:
            argument_results = json.load(f)
            
        argument_results = pd.DataFrame(argument_results)
        print(argument_results.fallacy_type.value_counts())
        
        with open(os.path.join(output_dir, 'claim_results.json')) as f:
            claim_results = json.load(f)
            
        claim_results = pd.DataFrame(claim_results)
        print(claim_results.fallacy_type.value_counts())