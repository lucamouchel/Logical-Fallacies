from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings
import json
from openai.error import RateLimitError, ServiceUnavailableError, APIError, APIConnectionError, Timeout, InvalidRequestError
import openai
import time 
import random
import pathlib

openai.api_key = '...'
warnings.filterwarnings("ignore")

data = pd.read_json('/Users/lucamouchel/Downloads/logicalfallacy/Logical-Fallacies/src/baselines/test_data.json')

examples = pd.read_json(os.path.join('/Users/lucamouchel/Downloads/logicalfallacy/Logical-Fallacies/data/argumentation/fallacies_arguments_support.json'))

def generate_prompt(topic, arg_type='support'):
    assert arg_type == 'support' or arg_type=='counter', 'your argument type does not fit the current data'

    text =  f"""You are given a topic.  
    Your task is to generate a {'supporting' if arg_type =='support' else 'counter'} argument in the context of the topic. 
    You must take into account logical fallacies. 
    Logical fallacies are error in reasoning that can make an argument seem valid but is actually invalid. You must generate a logical argument that addresses the topic.
    Some examples include: 
    - Forcing people to vote is wrong because it's not right to make them do something they shouldn't be forced to do. (Circular Reasoning)
    - I know four poor families. They are lazy drug addicts. Therefore, all poor people are lazy drug addicts. (Faulty Generalization)
    
    It should not be longer than 15 words. 
    return the following using this json format. Do not forget quotation marks:
    {"{"}
        "topic": {topic},
        "argument": <>
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
                    "temperature": 0.1,
                    "max_tokens": 200,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                    })
    
    
def generate_with_GPT(data_source: pd.DataFrame=data):
    
    output_dir = "/Users/lucamouchel/Downloads/logicalfallacy/Logical-Fallacies/src/baselines/gptdata"
    seen_topic = set()
    gpt_result = []
    i = 0
    for _, sample in tqdm(data_source.iterrows()):
        topic = sample['topic']
        if topic in seen_topic:
            continue
        else: 
            seen_topic.add(topic)
            
        if len(seen_topic) == 100:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = os.path.join(output_dir, f"gpt_zeroshot.json")
            with open(output_path, 'w') as json_file:
                json.dump(gpt_result, json_file, indent=4, sort_keys=False)

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
            
        prompt = generate_prompt(topic=topic, arg_type=stance)
        try:
            i+=1
            response = get_gpt_response(input=prompt)
            gpt_result.append(json.loads(response))
        except:
            # in case there is an unfortunate parsing error, then we just write everything we have, to not lose it
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = os.path.join(output_dir, f"gpt_zeroshot.json")
            with open(output_path, 'w') as json_file:
                json.dump(gpt_result, json_file, indent=4, sort_keys=False)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f"gpt_zeroshot.json")
    with open(output_path, 'w') as json_file:
        json.dump(gpt_result, json_file, indent=4, sort_keys=False)
            
        

generate_with_GPT(data)