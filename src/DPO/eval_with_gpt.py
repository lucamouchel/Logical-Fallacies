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
from env import OPENAI_API_KEY
import pathlib 
import argparse
from peft import AutoPeftModelForCausalLM

from utils import save_to, process_gpt_output, get_gpt_response, remove_incomplete_last_sentence, sanitize

warnings.filterwarnings("ignore")

data_dir = 'data/argumentation/'
output_dir = 'src/DPO/gpt_evaluation/'
openai.api_key = OPENAI_API_KEY


def generate(prompt: str, model, tokenizer):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=80, truncation=True).to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids=tokenized_prompt.input_ids,
                                attention_mask=tokenized_prompt.attention_mask,
                                max_new_tokens=30,
                                #pad_token_id=tokenizer.eos_token_id,
                                #top_p=0.5,
                                #temperature=1,
                                no_repeat_ngram_size=2,
                                do_sample=False)

    output_decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_decoded


def compare_models(stance, topic, arguments):
    possible_answers = ''
    for i, argument in enumerate(arguments):
        possible_answers += f"{i+1}. {argument}\n"
    
    
    prompt = f"""Which of these {stance} arguments is better for the topic: {topic}. Please consider the fact that some of these arguments might be in the form of logical fallacies. If one of the arguments is a logical fallacy, it should not be the best. 
    \nArguments: \n{possible_answers}\n
    If the arguments are equally good, return number 3.\nThe better argument is number:"""
    response = get_gpt_response(prompt, necessary_tokens=1, model='gpt-4')
    try:
        response = int(response)
        return response
    except:
        print("Invalid response. Please try again.")
        print(arguments)
        print()
        print(response)
        return 3

        
def get_gpt_feedback(topic, argument, type_='dpo'):
    fallacy_types = pd.read_csv('data/LOGIC/mappings.csv')['Original Name'].unique()
    fallacy_types = [f for f in fallacy_types if f != 'miscellaneous']
    s0 = f"Consider the following topic and argument:\nTopic: {topic}\nArgument: {argument}\n"
    s1 = f"""Out of all the following logical fallacy types\n{fallacy_types}\nwould you qualify this argument as one of these logical fallacies? If not - return "None".\n"""
    s2 = f"If yes, which logical fallacy type is it? Let fallacy_type be your answer.\n"
    s3 = "If an argument is not related to the topic, it should be considered as a fallacy of logic."
    s4 = f"""Your task is to complete the following json: {'{'} "type": "{type_}",\n "fallacy_type": <> {'}'}. If an argument does not make sense, set fallacy_type to "None"."""
    prompt = s0 + s1 + s2 + s3 + s4
    
    response = get_gpt_response(prompt, model='gpt-4')    
    response = process_gpt_output(response)
    if response is None:
        print("RETURNED NONE")
        response = {'type': type_, 'fallacy_type': 'None'}
    response['topic'] = topic
    response['argument'] = argument
    
    if argument==topic:
        response['fallacy_type'] = 'circular reasoning' ## SFT model often generates the given topic as an argument.
    return response



def evaluate(dataset, type_='sft', n=None):
    if type_=='sft':
        with open('results/sft_arguments.json', "r") as json_file:
            args = json.load(json_file)
    elif type_=='dpo':
        with open('results/dpo_arguments.json', "r") as json_file:
            args = json.load(json_file)

    f_rate = 0
    f_rates = {}
    data = []
    for i, entry in tqdm(dataset.iterrows()):
        topic = entry.topic
        y = args[i]
        feedback = get_gpt_feedback(topic, y, type_=type_)
        if feedback['fallacy_type']!='None' :
            f_rate+=1
        
        
        if feedback['fallacy_type'] in f_rates.keys():
            f_rates[feedback['fallacy_type']] += 1
        else:
            f_rates[feedback['fallacy_type']] = 1

        data.append(feedback)

    save_to(data, name=f'{type_}-f-rate.json', output_dir='results/')
    print(f_rates)
    print(f"f rate for {type_}:", f_rate)
    print("FALLACY TYPES")

    for k,v in f_rates.items():
        print(k.upper(), ':', v)

    
    
def calculate_win_rate(test_set):
    wins= {i : 0 for i in range(1, 4)} # +1 for ties
    
    
    with open('results/sft_arguments.json', "r") as json_file:
        sft_args = json.load(json_file)
    
    with open('results/dpo_arguments.json', "r") as json_file:
        dpo_args = json.load(json_file)
    
    for i, entry in tqdm(test_set.iterrows()):
        topic = entry.topic
        stance = 'supporting' if entry.label == 1 else 'counter'
        y_sft = sft_args[i]
        y_dpo = dpo_args[i]
        print(f"Generate a {stance} argument for the topic: {topic}")
    
        try:
            response = compare_models(stance, topic, [y_sft, y_dpo])
            wins[response] += 1
        except:
            save_to(wins, name='wins.json', output_dir='results/')
            continue
    save_to(wins, name='wins.json', output_dir='results/')
    

 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sft-path', default=None)
    parser.add_argument('--dpo-path', default=None)
    parser.add_argument('--num-iterations-made',default=0, type=int, help='Number of iterations made for the selft instruction')
    parser.add_argument('--eval-type', default='win-rate', help='Type of evaluation to perform. Options: win-rate, fallacy-count')
    parser.add_argument('--generate-and-save', default='true', type=str)
    return parser.parse_args()

def main(): 
    args = parse_args()
    ref_model_dir = args.sft_path
    dpo_model_dir = args.dpo_path
    num_iterations = args.num_iterations_made
    

    
    is_encoder_decoder = 't5' in dpo_model_dir if dpo_model_dir else 't5' in ref_model_dir
    test_set = pd.read_json('data/argumentation/test_cckg.json')
    test_set = test_set.reset_index()
    
    if args.generate_and_save=='false':
        print("EVALUTATION MODE")
        if args.eval_type == 'win-rate':
            calculate_win_rate(test_set)
        elif args.eval_type == 'fallacy-count':
            evaluate(test_set, type_='sft' if ref_model_dir else 'dpo')

        exit()
    
    models = []
    if is_encoder_decoder:
        if ref_model_dir:
            sft_model = transformers.T5ForConditionalGeneration.from_pretrained(ref_model_dir)
            models.append(sft_model)
        if dpo_model_dir:
            dpo_model = transformers.T5ForConditionalGeneration.from_pretrained(dpo_model_dir)
            models.append(dpo_model)
        
        for i in range(1, num_iterations + 1):
            model = transformers.T5ForConditionalGeneration.from_pretrained(dpo_model_dir + f'/iteration{i}')
            models.append(model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(ref_model_dir)
    else:
        if ref_model_dir:
            sft_model = AutoPeftModelForCausalLM.from_pretrained(ref_model_dir, device_map='auto')
            models.append(sft_model)
        if dpo_model_dir:
            dpo_model = AutoPeftModelForCausalLM.from_pretrained(dpo_model_dir, device_map='auto')
            models.append(dpo_model)

        for i in range(1, num_iterations + 1):
            model = transformers.AutoModelForCausalLM.from_pretrained(dpo_model_dir + f'/iteration{i}', device_map='auto')
            models.append(model)
            
        tokenizer = transformers.AutoTokenizer.from_pretrained(ref_model_dir)


    if args.generate_and_save=='false':
        print("EVALUTATION MODE")
        if args.eval_type == 'win-rate':
            calculate_win_rate(models, test_set, tokenizer)
        elif args.eval_type == 'fallacy-count':
            evaluate(test_set, tokenizer, type_='sft' if ref_model_dir else 'dpo')

        exit()
    

    for j, model in enumerate(models):
        model_arguments = []
        print(model.name_or_path)
        for i, entry in tqdm(test_set.iterrows()):
            topic = entry.topic
            stance = 'SUPPORTING' if entry.label == 1 else 'COUNTER'
            prompt = f"<s> [INST] ###Prompt:  Generate a {stance} argument for the topic: {topic} [INST]\n### Argument: "
            generated = generate(prompt, model, tokenizer)
             #if prompt in generated:
             #   generated = generated[len(prompt) + 1:]
            #generated = sanitize(remove_incomplete_last_sentence(generated))
            
        
            generated = generated[generated.find('### Argument: ')+len('### Argument: '):].strip()
            model_arguments.append(generated)
        
        if j == 0:
            save_to(model_arguments, name='sft_arguments.json', output_dir='results/')
        elif j == 1:
            save_to(model_arguments, name='dpo_arguments.json', output_dir='results/')
        elif j>1:
            save_to(model_arguments, name =f'dpo_arguments_iteration{j}.json', output_dir='results')




    # df_sft = pd.read_json('src/DPO/gpt_evaluation/sft_test_responses.json')
    # df_dpo = pd.read_json('src/DPO/gpt_evaluation/dpo_test_responses.json')
    
    # fig, ax = plt.subplots(1,2, figsize=(20, 10), sharey=True)

    # df = df_sft.merge(df_dpo, on=['topic', 'real_stance'], suffixes=('_sft', '_dpo'))
    
    # fallacy_counts_sft = df_sft.fallacy_type.value_counts()
    # fallacy_counts_sft.pop('None')
    # ax[0].bar([f[0] for f in fallacy_counts_sft.items()], [f[1] for f in fallacy_counts_sft.items()])
    # print("Num fallacies in SFT: ", len(df_sft[df_sft.fallacy_type != 'None']))
    
    
    # fallacy_counts_dpo = df_dpo.fallacy_type.value_counts()
    # fallacy_counts_dpo.pop('None')
    # ax[1].bar([f[0] for f in fallacy_counts_dpo.items()], [f[1] for f in fallacy_counts_dpo.items()])
    # print("Num fallacies in DPO: ", len(df_dpo[df_dpo.fallacy_type != 'None']))
    
    
    # plt.savefig('src/DPO/gpt_evaluation/fig.png')
    
if __name__ == "__main__":
    main()

