
import json
import argparse
import transformers
import torch
import gc 
import pandas as pd 
from datasets import Dataset
from tqdm import tqdm
from trl import DPOTrainer, ORPOConfig, ORPOTrainer
import utils
from collections import namedtuple


def main():
    model_path = arguments.model_path
    if model_path[-1] != '/':
        model_path += '/'
    ## dda
    args = dict(json.load(open(model_path+'args.json', 'r')))
    args = namedtuple('DotDict', args.keys())(**args)    
    
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')        
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    fallacy_classifier = transformers.AutoModelForSequenceClassification.from_pretrained(arguments.clf_path, num_labels=2)
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
            is_encoder_decoder=False)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        print("ITERATION ", iteration, " AVERAGE : ", avg)
        iteration += 1
        if avg <= arguments.fallacy_threshold:
            break

        model = transformers.AutoModelForCausalLM.from_pretrained(new_output_dir, device_map='auto')
                
        
    print("FINAL MODEL saved at ", new_output_dir)

def map_data(example):
    return {
        'prompt': '<s> [INST] ### Prompt: ' + example['prompt'] + f" [/INST]\n### Argument:" ,
        'chosen': example['chosen'] + ' </s>',
        'rejected': example['rejected'] + ' </s>'
    }
    
    
def run_self_reward(args, model, clf, clf_tokenizer, tokenizer, iteration_number, is_encoder_decoder):
    print("Running self reward loop -- iteration", iteration_number)
    clf.eval()
    def fallacy_proba(y):
        input_ids = clf_tokenizer.encode(y, add_special_tokens=True, return_tensors='pt')
        with torch.no_grad():
            out = clf(input_ids=input_ids)
            
        logits = out.logits
        fallacy_logit = logits[0][1]        
        fallacy_proba = torch.sigmoid(fallacy_logit).item()
        return fallacy_proba
    
    GENERATION_KWARGS = {'max_new_tokens': 30, 'no_repeat_ngram_size': 2, 'do_sample': True, 'min_new_tokens': 5, 'top_p': 0.75}    
    def generate(prompt: str, model, tokenizer, n):
        """Main function for text generation."""
        tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=128, truncation=True).to(model.device)
        with torch.no_grad():
            output = model.generate(**tokenized_prompt,
                                    **GENERATION_KWARGS,
                                    num_return_sequences=n,
                                    pad_token_id=tokenizer.eos_token_id)
        output_decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        return output_decoded

    with open('data/dpo/arguments/dev.json', 'r') as f:
        test = json.load(f)

    n = 4 ## generate 4 responses for each prompt
    
    test = test[:100]
    new_dataset = []
    total_probas = []
    print("Sampling responses and generating fallacy probabilities.")
    
    for sample in tqdm(test, total=len(test)):
        chosen = sample['chosen']
        prompt = sample['prompt']
        if 'supporting' in prompt:
            prompt = prompt.replace('supporting', 'SUPPORTING')
        elif 'counter' in prompt:
            prompt = prompt.replace('counter', 'COUNTER')
            
        prompt = '<s> [INST] ### Prompt: ' + prompt + " [/INST]\n### Argument: "
        ys = generate(prompt, model, tokenizer, n) 
        for y in ys:
            y = y.split('### Argument:')[-1].strip()
            proba = fallacy_proba(y)
            if proba > 0.5:
                sample = {'prompt': prompt, 'chosen': chosen, 'rejected': y}
                new_dataset.append(sample)
                print(sample)
            total_probas.append(proba)

    print("DATASET SIZE: ", len(new_dataset))
    avg = sum(total_probas)
    avg = avg/(n*len(new_dataset))
    print('AVERAGE FALLACY PROBA: ', avg)

    if avg <= arguments.fallacy_threshold:
        print("THRESHOLD Validated!! AVG:", avg)
        return avg, args.output_dir + f'/iteration{iteration_number-1}'
 
    training_data = Dataset.from_dict(pd.DataFrame(new_dataset))
    training_data.map(map_data)
    training_args = utils.get_training_args(args)
    
    optimization_algorithm = arguments.optim_algorithm
    
    if optimization_algorithm == 'DPO':
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=model,
            beta=args.beta,
            args=training_args,
            train_dataset=training_data,
            is_encoder_decoder=is_encoder_decoder,
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_prompt_length=args.max_length,
            max_target_length=args.max_length,
            generate_during_eval=True)

        dpo_trainer.train()
        new_output_dir = args.output_dir + f'/iteration{iteration_number}'
        dpo_trainer.save_model(new_output_dir)
        
        del dpo_trainer
        torch.cuda.empty_cache()
        print("SAVED MODEL at ", new_output_dir)
        
    elif arguments.optim_algorithm == 'ORPO':
        orpo_config = ORPOConfig(
            **training_args.to_dict(),
            beta=args.beta, ## 0.1
            max_prompt_length=150,
            max_length=args.max_length,
            is_encoder_decoder=False,
        )
        
        orpo_trainer = ORPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=orpo_config,
            train_dataset=training_data,
        )
        orpo_trainer.train()
        new_output_dir = args.output_dir + f'/iteration{iteration_number}'
        print("SAVING MODEL at ", new_output_dir)
        orpo_trainer.save_model(new_output_dir)

        
        
    
    return avg, new_output_dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--clf-path', default='models/fallacy_clf/howey_electra-base-mnli')
    parser.add_argument('--fallacy-threshold', default=0.13, type=float)
    parser.add_argument('--optim-algorithm', required=True, help='DPO, ORPO')
    return parser.parse_args()

if __name__ == '__main__':
    arguments = parse_args()
    main()