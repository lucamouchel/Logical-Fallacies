from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import os
from tqdm import tqdm
import warnings
from data_loader import load_data
import json
from openai.error import RateLimitError, ServiceUnavailableError, APIError, APIConnectionError, Timeout, InvalidRequestError
import openai
import time 
import random
import pathlib

warnings.filterwarnings("ignore")

data_dir = 'data/argumentation/'
train_cckg, dev_cckg, test_cckg = load_data('cckg')
train_cckg['fallacy_type'] = 'No Fallacy'
dev_cckg['fallacy_type'] = 'No Fallacy'
test_cckg['fallacy_type'] = 'No Fallacy'

examples = pd.read_json(os.path.join(data_dir,'fallacies_arguments_support.json'))
counts = pd.read_csv('data/LOGIC/edu_all.csv').updated_label.value_counts()
fallacy_distributions = (counts/sum(counts))

def generate_prompt(topic, fallacy_type, arg_type='support'):
    assert arg_type == 'support' or arg_type=='counter', 'your argument type does not fit the current data'
    assert fallacy_type in counts.keys(), "Fallacy type not in the current data!"
    data = examples[fallacy_type]
    args = [data[f"exampleArg1{arg_type}"], data[f"exampleArg2{arg_type}"]]
    fallacies = [data['example1'], data['example2']]
     
    text =  f"""You are given a topic.  
    Your task is to generate a {'supporting' if arg_type =='support' else 'counter'} argument in the form of a {fallacy_type} logical fallacy in the context of the topic. 
    It should not be longer than 25 words. 
    
    {fallacy_type} fallacy is defined as: {examples[fallacy_type]['definition']}
    examples of {fallacy_type} fallacy are: 
    {fallacies[0]}
    {fallacies[1]}
        
    Here is an example of a supporting {fallacy_type} fallacy: 
    {args[0]}
    
    return the following using this json format. Do not forget quotation marks:
    {"{"}
        "topic": {topic},
        "fallacy type": {fallacy_type},
        "{fallacy_type} fallacy {arg_type}": ...
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
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                    })
    
    
def generate_with_GPT(data_source: pd.DataFrame=train_cckg, fallacy_distributions: dict=fallacy_distributions):
    """
    Augments the data source with fallacies using chatgpt

    Args:
        data_source (pd.DataFrame): the data to augment - eg augmenting the training/dev/test data
        fallacy_distributions (dict): dict with keys (fallacy types) and values (probability of picking this type)
        
        Logically, the fallacy distributions should not be uniform because some fallacies occur more often than others
        By default, the distributions follow the distributions in the LOGIC dataset!!
        
        top 5 occuring fallacies:
        faulty generalization     0.179853
        ad hominem                0.123165
        ad populum                0.094617
        false causality           0.088091
        circular reasoning        0.069739
    """
    
    output_dir = "data/generated/"
    seen_topic = set()
    gpt_result = []
    i = 0
    for _, sample in tqdm(data_source.iterrows()):
        topic = sample['topic']
        if topic in seen_topic:
            continue
        else: seen_topic.add(topic)
        data_for_topic = data_source[data_source.topic==topic]
        num_topic_occurences = data_for_topic.shape[0]
        if num_topic_occurences > 1:
            majority_stance = data_for_topic['label'].sum()
            if majority_stance < 0:
                stance = 'counter'
            elif majority_stance > 0:
                stance = 'support'
            else: 
                stance = random.choice(['support', 'counter'])
        else:
            stance = 'support' if sample['label'] == 1 else 'counter'
            #print(stance)
            
        chosen_fallacy_types = np.random.choice(fallacy_distributions.keys(), p=fallacy_distributions.values, size=4, replace=False)
        for fallacy_type in (chosen_fallacy_types):
            if fallacy_type=='miscellaneous':
                continue
            prompt = generate_prompt(topic=topic, fallacy_type=fallacy_type, arg_type=stance)
            try:
                i+=1
                response = get_gpt_response(input=prompt)
                gpt_result.append(json.loads(response))
            except:
                # in case there is an unfortunate parsing error, then we just write everything we have, to not lose it
                pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
                output_path = os.path.join(output_dir, f"train.json")
                with open(output_path, 'w') as json_file:
                    json.dump(gpt_result, json_file, indent=4, sort_keys=False)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f"train.json")
    with open(output_path, 'w') as json_file:
        json.dump(gpt_result, json_file, indent=4, sort_keys=False)
            
        
generate_with_GPT(train_cckg)
