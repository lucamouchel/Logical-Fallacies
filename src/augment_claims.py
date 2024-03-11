from matplotlib.pyplot import cla
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

data_dir = 'data/claims/'

train, dev = load_claim_data('train'), load_claim_data('dev')
counts = pd.read_csv('data/LOGIC/edu_all.csv').updated_label.value_counts()
fallacy_distributions = (counts/sum(counts))
openai.api_key = OPENAI_API_KEY
output_dir = 'data/generated/claims/'

#examples = pd.read_json(os.path.join(data_dir,'fallacies_arguments_support.json'))

def generate_prompt(evidence: list, original_claim, claim_type):
    if len(evidence) > 3: evidence = evidence[:3]    
    text = f"""Given the following evidence, generate a {claim_type} claim in the form of a logical fallacy. The original claim is {original_claim}. You can base your fallacy claim on the original one. 
        Select the most appropriate fallacy type from the following list: {list(counts.keys())}. Let gpt_fallacy be your answer.
    Evidence: {evidence}.
    Your fallacy claim should not exceed 25 words. Let gpt_claim be your claim. Complete the following json:
    {"{"}
        "fallacy type": gpt_fallacy,
        "fallacy claim": gpt_claim
    {"}"}
    """
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
                "temperature": 0.0,
                "max_tokens": 150,
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
                print("Failed to extract a valid dictionary from the text.")
                return None
        else:
            print("No dictionary found in the text.")
            return None
        

def save_to(data, name):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, name)
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, sort_keys=False)


def generate_with_GPT(data, split='train'):
    gpt_result = []

    for _, sample in tqdm(data.iterrows()):
        evidence = sample['evidence']
        claim_type = sample['label']
        
        claim_type = 'support' if claim_type == 'SUPPORTS' else 'refuting'
        original_claim = sample['claim']
        
        prompt = generate_prompt(evidence, original_claim=original_claim, claim_type=claim_type)
        try:
            gpt_response = get_gpt_response(prompt)
            gpt_response = process_gpt_output(gpt_response)
            if gpt_response is None:
                continue
            gpt_response['evidence'] = evidence
            gpt_response['original claim'] = original_claim
            gpt_response['claim type'] = claim_type
            gpt_result.append(gpt_response)
        except:
            save_to(gpt_result, f'{split}_generated.json')
            continue
        
        
    save_to(gpt_result, f'{split}_generated.json')
    
data = load_claim_data_sampled('train', n=8000)
generate_with_GPT(data, split='train')