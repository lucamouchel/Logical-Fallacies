import numpy as np
import json
import argparse
import transformers
import torch
import gc 
import pandas as pd 
from datasets import Dataset
from tqdm import tqdm
from trl import DPOTrainer
import utils
from collections import namedtuple
import nltk

def main():
    dpo_model_path = arguments.dpo_model_path
    if dpo_model_path[-1] != '/':
        dpo_model_path += '/'
    
    args = dict(json.load(open(dpo_model_path+'args.json', 'r')))
    args = namedtuple('DotDict', args.keys())(**args)    
    
    if 't5' in args.model_name.lower():
        is_encoder_decoder=True
        model = transformers.T5ForConditionalGeneration.from_pretrained(dpo_model_path).to('cuda')
    elif 'llama' in args.model_name.lower():
        is_encoder_decoder=False
        model = transformers.LlamaForCausalLM.from_pretrained(dpo_model_path).to('cuda') 
        tokenizer = transformers.LlamaTokenizer.from_pretrained(dpo_model_path)
    else:
        is_encoder_decoder=False
        model = transformers.AutoModelForCausalLM.from_pretrained(dpo_model_path).to('cuda')
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(dpo_model_path)

    fallacy_classifier = transformers.AutoModelForSequenceClassification.from_pretrained(arguments.clf_path, num_labels=2).to('cuda')
    clf_tokenizer = transformers.AutoTokenizer.from_pretrained(arguments.clf_path)

    iteration = 1
    while True:
        avg, new_output_dir = run_self_reward(
            args=args, 
            model=model, 
            clf=fallacy_classifier, 
            clf_tokenizer=clf_tokenizer, 
            tokenizer=tokenizer, 
            iteration_number=iteration,
            is_encoder_decoder=is_encoder_decoder,
            dpo_model_path=dpo_model_path)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        print("ITERATION ", iteration, " AVERAGE : ", avg)
        iteration += 1
        if avg <= arguments.fallacy_threshold:
            break
        
        if 't5' in args.model_name.lower():
            model = transformers.T5ForConditionalGeneration.from_pretrained(new_output_dir).to('cuda')
        elif 'llama' in args.model_name.lower():
            model = transformers.LlamaForCausalLM.from_pretrained(new_output_dir).to('cuda') 
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(new_output_dir).to('cuda')
        
        
    print("FINAL MODEL saved at ", new_output_dir)

        
def run_self_reward(args, model, clf, clf_tokenizer, tokenizer, iteration_number, is_encoder_decoder, dpo_model_path):
    print("Running self reward loop -- iteration", iteration_number)
    clf.eval()
    model.eval()
    causal_LM_FOR_PROBA = transformers.AutoModelForCausalLM.from_pretrained('google/gemma-2b').to('cuda')
    def fallacy_proba(y):
        input_ids = clf_tokenizer.encode(y, add_special_tokens=True, return_tensors='pt').to('cuda')

        with torch.no_grad():
            out = clf(input_ids=input_ids)
            
        logits = out.logits
        fallacy_logit = logits[0][1]        
        fallacy_proba = torch.sigmoid(fallacy_logit).item()
        return fallacy_proba
        
    def fallacy_proba_with_prompts(y, model, tokenizer):
        
        prompt = f"""
        Given the following argument, determine how likely it is a logical fallacy. A logical fallacy is an error in reasoning that undermines the validity of an argument or conclusion. 
        The possible types of fallacies are the following: [faulty generalization, false causality, circular reasoning, ad populum, ad hominem, fallacy of logic, appeal to emotion, false dilemma, equivocation, fallacy of extension, fallacy of relevance, fallacy of credibility, intentional]
        You must assign a probability between 0 and 1, from least to most likely of being a fallacy.
        Argument: {y}
        Probability: <RESPONSE>
        """
        tokenized_response = tokenizer(prompt, return_tensors='pt').to('cuda')
    
        with torch.no_grad():
            output = causal_LM_FOR_PROBA.generate(input_ids=tokenized_response.input_ids, 
                                    attention_mask=tokenized_response.attention_mask,
                                    max_new_tokens=2,
                                    pad_token_id=tokenizer.pad_token_id)
            
            output_decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            if prompt in output_decoded: 
                output_decoded = output_decoded[len(prompt) + 1: ]
            try:
                return float(output_decoded)
            except:
                print("ERROR: ", output_decoded)
                return 0.5
            

        
    def generate(prompt: str, model, tokenizer, n=4):
        """Main function for text generation."""
        tokenized_prompt = tokenizer(prompt, return_tensors='pt').to('cuda')
        with torch.no_grad():
            output = model.generate(input_ids=tokenized_prompt.input_ids, 
                                attention_mask=tokenized_prompt.attention_mask,
                                max_new_tokens=30,
                                top_p=0.95,
                                temperature=1,
                                repetition_penalty=1.5,
                                num_return_sequences=n,
                                pad_token_id=tokenizer.eos_token_id,                                  
                                do_sample=True)
        
        output_decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        return output_decoded

    test = utils.get_data(args.data_dir, split='test', return_type='df')
    df = test
    prompts = df.prompt.unique() ### only 225 for now bc of memory constraints
    n = 4 ## generate 4 responses for each prompt

    new_dataset = {'prompt': [], 'chosen': [], 'rejected': []}
    total_probas = []
    dataset_size=0
    print("Sampling responses and generating fallacy probabilities.")
    kept_probas=[]
    for i, item in tqdm(df.iterrows()):
        golden = item.chosen
        prompt = item.prompt
        if prompt[-1] not in ['.', '?', '!']:
            prompt+='.'

        prompt += ' -- <RESPONSE>'
        ys = []
        generated = generate(prompt, model, tokenizer, n=n)
        for y in generated:
            if prompt in y:
                y = y[len(prompt)+1:] #y.replace(prompt, '')
            y = utils.sanitize(utils.remove_incomplete_last_sentence(y))
            ys.append(y)


        probas = []
        for y in ys:
            proba = fallacy_proba_with_prompts(y, model, tokenizer)
            print(proba)
            probas.append(proba)

        total_probas.append(probas)
        arr = sorted(zip(ys, probas), key=lambda x: x[1])
        for i in range(len(arr)//2):
            ### pair the highest proba fallacy with the lowest one
            y_w = arr[i][0]
            y_l = arr[-(i+1)][0]
            if arr[i][1] < 0.5 and arr[-(i+1)][1] > 0.5:
                dataset_size+=1
                kept_probas.append(arr[i][1])
                kept_probas.append(arr[-(i+1)][1])
                new_dataset['prompt'].append(prompt)
                new_dataset['chosen'].append(y_w)
                new_dataset['rejected'].append(y_l)


    print("DATASET SIZE: ", len(new_dataset['chosen']))

    avg = 0    
    for arr in total_probas:
        for proba in arr:
            avg += proba
                
    avg = avg/(dataset_size)
    
    avg = sum(kept_probas) / len(kept_probas)


    print("AVERAGE : ", avg)
    print(('\n'))
    if avg <= arguments.fallacy_threshold:
        print("THRESHOLD Validated!! AVG:", avg)
        return avg, args.output_dir + f'/iteration{iteration_number-1}'

    print(model.name_or_path)
    
    for i in range(10, 20):
         print(new_dataset['prompt'][i])
         print(new_dataset['chosen'][i])
         print(new_dataset['rejected'][i])
         print('\n')

    training_data = Dataset.from_dict(pd.DataFrame(new_dataset))
    model = transformers.AutoModelForCausalLM.from_pretrained(dpo_model_path, device_map='auto')
    dpo_trainer = DPOTrainer(
        model=model,
        ref_