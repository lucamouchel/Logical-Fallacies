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
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    
    prompt = f"""Which of these {stance} arguments is better for the topic: {topic}.
    Please consider the fact that some of these arguments might be in the form of logical fallacies. 
    If one of the arguments is a logical fallacy, it should not be the best. 
    \nArguments: {possible_answers}\n
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

        
def get_gpt_feedback(stance, topic, argument, type_='dpo'):
    fallacy_types = pd.read_csv('data/LOGIC/mappings.csv')['Original Name'].unique()
    fallacy_types = [f for f in fallacy_types if f != 'miscellaneous']
    s0 = f"Consider the following argument: {argument}\n"
    s1 = f"""Out of all the following logical fallacy types\n{fallacy_types}\nwould you qualify this argument as one of these logical fallacies? If not - return "None".\n"""
    s2 = f"If yes, which logical fallacy type is it? Let fallacy_type be your answer.\n"
    s3 = f"""Your task is to complete the following json: {'{'} "type": "{type_}",\n "fallacy_type": <> {'}'}. If an argument does not make sense, set fallacy_type to "None"."""
    prompt = s0 + s1 + s2 + s3 
    
    response = get_gpt_response(prompt, model='gpt-4')    
    response = process_gpt_output(response)
    if response is None:
        print("RETURNED NONE")
        response = {'type': type_, 'gpt_stance': 'supporting', 'fallacy_type': 'None'}
    response['topic'] = topic
    response['argument'] = argument
    
    if argument==topic:
        response['fallacy_type'] = 'circular reasoning' ## SFT model often generates the given topic as an argument.
    return response



def evaluate(model, dataset, tokenizer, type_='sft', n=None):
    with open('results/sft_arguments.json', "r") as json_file:
        sft_args = json.load(json_file)

    with open('results/dpo_arguments.json', "r") as json_file:
        dpo_args = json.load(json_file)


    sft_f_rate = 0
    dpo_f_rate = 0
    sft_data = []
    dpo_data = []
    for i, entry in tqdm(dataset.iterrows()):
        topic = entry.topic
        stance = 'supporting' if entry.label == 1 else 'counter'
        y_sft = sft_args[i]
        y_dpo = dpo_args[i]
        sft_gpt = get_gpt_feedback(stance, topic, y_sft, type_='sft')
        
        dpo_gpt = get_gpt_feedback(stance, topic, y_dpo, type_='dpo')
        
        if sft_gpt['fallacy_type']!='None' :
            sft_f_rate+=1
            print('SFT FALLACY')
            print(sft_gpt['argument'])
        if dpo_gpt['fallacy_type']!='None':
            dpo_f_rate+=1
            print("DPO FALLACY")
            print(dpo_gpt['argument'])
        sft_data.append(sft_gpt)
        dpo_data.append(dpo_gpt)


    save_to(sft_data, name='sft-f-rate.json', output_dir='results/')
    save_to(dpo_data, name='dpo-f-rate.json', output_dir='results/')
    print("SFT f rate:", sft_f_rate)
    print("DPO f rate:", dpo_f_rate)
    
    
def calculate_win_rate(models, test_set, tokenizer):
    wins= {i : 0 for i in range(1, len(models)+1+1)} # +1 for ties
    
    
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
    parser.add_argument('--sft-path', default='models/sft_flan-t5-large_trl')
    parser.add_argument('--dpo-path', default='models/dpo_flan-t5-large_trl')
    parser.add_argument('--num-iterations-made',default=0, type=int, help='Number of iterations made for the selft instruction')
    parser.add_argument('--eval-type', default='win-rate', help='Type of evaluation to perform. Options: win-rate, fallacy-count')
    parser.add_argument('--generate-and-save', default=True, type=bool)
    return parser.parse_args()

def main(): 
    args = parse_args()
    ref_model_dir = args.sft_path
    dpo_model_dir = args.dpo_path
    num_iterations = args.num_iterations_made
    
    
    is_encoder_decoder = 't5' in dpo_model_dir
    test_set = pd.read_json('data/argumentation/test_cckg.json')
    test_set = test_set[:100].reset_index()
    models = []
    if is_encoder_decoder:
        sft_model = transformers.T5ForConditionalGeneration.from_pretrained(ref_model_dir)
        dpo_model = transformers.T5ForConditionalGeneration.from_pretrained(dpo_model_dir)
        models.extend([sft_model, dpo_model])
        for i in range(1, num_iterations + 1):
            model = transformers.T5ForConditionalGeneration.from_pretrained(dpo_model_dir + f'/iteration{i}')
            models.append(model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(ref_model_dir)
    else:
        sft_model = transformers.AutoModelForCausalLM.from_pretrained(ref_model_dir, device_map='auto')
        dpo_model = transformers.AutoModelForCausalLM.from_pretrained(dpo_model_dir, device_map='auto')
        models.extend([sft_model, dpo_model])
        for i in range(1, num_iterations + 1):
            model = transformers.AutoModelForCausalLM.from_pretrained(dpo_model_dir + f'/iteration{i}', device_map='auto')
            models.append(model)
            
        tokenizer = transformers.AutoTokenizer.from_pretrained(ref_model_dir)


    if not generate_and_eval:
        if args.eval_type == 'win-rate':
            calculate_win_rate(models, test_set, tokenizer)
        elif args.eval_type == 'fallacy-count':
            evaluate(sft_model, test_set, tokenizer, type_='sft')

    exit()
    

    for j, model in enumerate(models):
        model_arguments = []
        print(model.name_or_path)
        for i, entry in tqdm(test_set.iterrows()):
            topic = entry.topic
            stance = 'SUPPORTING' if entry.label == 1 else 'COUNTER'
            prompt = f"<s> [INST] Generate a {stance} argument for the topic: {topic} [INST]\n### Argument: "
            generated = generate(prompt, model, tokenizer)
            if prompt in generated:
                generated = generated[len(prompt) + 1:]
        
            generated = sanitize(remove_incomplete_last_sentence(generated))
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

