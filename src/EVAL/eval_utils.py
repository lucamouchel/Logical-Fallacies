
import pandas as pd
from tqdm import tqdm
import warnings
import json
import openai
from env import OPENAI_API_KEY
from utils import save_to, process_gpt_output, get_gpt_response
warnings.filterwarnings("ignore")
openai.api_key = OPENAI_API_KEY


def get_gpt_feedback(topic, argument, stance, type_='dpo'):
    fallacy_types = pd.read_csv('data/LOGIC/mappings.csv')['Original Name'].unique()
    fallacy_types = [f for f in fallacy_types if f != 'miscellaneous']
    s0 = f"Consider the following topic and the {stance} argument:\nTopic: {topic}\nArgument: {argument}\n"
    s1 = f"""Out of all the fallacy types: {fallacy_types}, would you qualify this argument as a logical fallacy? If not - return "None"."""
    s2 = f"If the given argument is a logical fallacy, which type is it? Let fallacy_type be your answer. You must be very sure about your answer, if you are not 100% sure of the fallacy type you have chosen, let fallacy_type be 'None'.\n" 
    s3 = f"""Your task is to complete the following json: {'{'} "type": "{type_}",\n "fallacy_type": <> {'}'}."""
    prompt = s0 + s1 + s2 + s3
    response = process_gpt_output(get_gpt_response(prompt, model='gpt-4o'))
    if response is None:
        print("RETURNED NONE")
        response = {'type': type_, 'fallacy_type': 'None'}
        
    response['topic'] = topic
    response['argument'] = argument
    return response

def evaluate(dataset, type_, model_name):
    f_rates = {}
    data = []

    with open(f'results/{model_name}/{type_}_args.json', 'r') as f:
        arguments = json.load(f)


    for i, entry in tqdm(dataset[:200].iterrows(), total=len(dataset)):
        topic = entry.topic
        stance = 'supporting' if entry.label==1 else 'counter'
        y = arguments[i]
        print(y)
        feedback = get_gpt_feedback(topic, y, stance=stance, type_=type_)

    
        if feedback['fallacy_type'] in f_rates.keys():
            f_rates[feedback['fallacy_type']] += 1
        else:
            f_rates[feedback['fallacy_type']] = 1
        data.append(feedback)
        
        if i % 20 == 0:
            print("Fallacy rates so far: ") 
            print(f_rates)

    save_to(data, name=f'f-rate.json', output_dir=f'models_rebuttal/results/{model_name}/{type_}/')
    save_to(f_rates, name=f'fallacy_counts.json', output_dir=f'models_rebuttal/results/{model_name}/{type_}/')


def evaluate_from_file(dataset, type_, model_name):
    f_rates = {}
    data = []

    with open(f'results/{model_name}/{type_}_args.json', 'r') as f:
        arguments = json.load(f) 

    for i, entry in tqdm(dataset.iterrows(), total=len(dataset)):
        topic = entry.topic
        stance = 'supporting' if entry.label==1 else 'counter'

        y = arguments[i]
        if i % 20 == 0:
            print("Fallacy rate so far: ", f_rate) 
            print(f_rates)
        feedback = get_gpt_feedback(topic, y, stance=stance, type_=type_)
        if feedback['fallacy_type']!='None' :
            f_rate+=1
        if feedback['fallacy_type'] in f_rates.keys():
            f_rates[feedback['fallacy_type']] += 1
        else:
            f_rates[feedback['fallacy_type']] = 1
        data.append(feedback)

    save_to(data, name=f'f-rate.json', output_dir=f'results/{model_name}/{type_}/')
    save_to(f_rates, name=f'fallacy_counts.json', output_dir=f'results/{model_name}/{type_}/')
    

def evaluate_out_of_domain(type_, model_name):
    f_rates = {}
    data = []
    
    with open(f'src/TRANSFER/generated_debates_custom_cpo.json', 'r') as f:
        data = json.load(f) 
    
    results = []

    for entry in tqdm(data):
        try:
            topic = entry['topic']
            stance = entry['stance']
            y = entry['generated']
            feedback = get_gpt_feedback(topic, y, stance=stance, type_=type_)
            if feedback['fallacy_type'] in f_rates.keys():
                f_rates[feedback['fallacy_type']] += 1
            else:
                f_rates[feedback['fallacy_type']] = 1
            results.append(feedback)
        except:
            continue

    save_to(results, name=f'f-rate.json', output_dir=f'results/out_of_domain/{model_name}/{type_}/')    
    save_to(f_rates, name=f'fallacy_counts.json', output_dir=f'results/out_of_domain/{model_name}/{type_}/')
 

