
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


def main():
    dpo_model_path = arguments.dpo_model_path
    if dpo_model_path[-1] != '/':
        dpo_model_path += '/'
    
    args = dict(json.load(open(dpo_model_path+'args.json', 'r')))
    args = namedtuple('DotDict', args.keys())(**args)    
    
    is_encoder_decoder = 't5' in args.model_name
    if is_encoder_decoder:
        model = transformers.T5ForConditionalGeneration.from_pretrained(dpo_model_path)
    else: 
        model = transformers.AutoModelForCausalLM.from_pretrained(dpo_model_path, device_map='auto')
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(dpo_model_path)

    fallacy_classifier = transformers.AutoModelForSequenceClassification.from_pretrained(arguments.clf_path, num_labels=2, device_map='auto')
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
            is_encoder_decoder=is_encoder_decoder)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        print("ITERATION ", iteration, " AVERAGE : ", avg)
        iteration += 1
        if avg <= arguments.fallacy_threshold:
            break
        
        if is_encoder_decoder:
            model = transformers.T5ForConditionalGeneration.from_pretrained(new_output_dir)
        else: 
            model = transformers.AutoModelForCausalLM.from_pretrained(new_output_dir, device_map='auto')
                
        
    print("FINAL MODEL saved at ", new_output_dir)

        
def run_self_reward(args, model, clf, clf_tokenizer, tokenizer, iteration_number, is_encoder_decoder):
    print("Running self reward loop -- iteration", iteration_number)
    clf.eval()
    model.eval()

    def fallacy_proba(y):
        input_ids = clf_tokenizer.encode(y, add_special_tokens=True, return_tensors='pt')
        with torch.no_grad():
            out = clf(input_ids=input_ids)
            
        logits = out.logits
        fallacy_logit = logits[0][1]        
        fallacy_proba = torch.sigmoid(fallacy_logit).item()
        return fallacy_proba
        
    def generate(prompt: str, model, tokenizer, n):
        """Main function for text generation."""
        tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=256, truncation=True).to(model.device)
        with torch.no_grad():
            output = model.generate(input_ids=tokenized_prompt.input_ids,
                                attention_mask=tokenized_prompt.attention_mask,
                                max_new_tokens=30,
                                pad_token_id=tokenizer.eos_token_id,
                                #top_p=0.5,
                                #temperature=1,
                                num_return_sequences=n,
                                no_repeat_ngram_size=2,
                                do_sample=True)
        output_decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        return output_decoded

    with open('data/dpo/arguments/dev.json', 'r') as f:
        test = json.load(f)

    n = 4 ## generate 4 responses for each prompt
    
    test = test[:5]
    new_dataset = []
    total_probas = []
    print("Sampling responses and generating fallacy probabilities.")
    
    for sample in tqdm(test, total=len(test)):
        chosen = sample['chosen']
        prompt = sample['prompt']

        ys = generate(prompt, model, tokenizer, n)
        print(ys)
        for y in ys:
            proba = fallacy_proba(y)
            print(proba)
            if proba > 0.5:
                sample = {'prompt': prompt, 'chosen': chosen, 'rejected': y}
                new_dataset.append(sample)
                total_probas.append(proba)
                
            total_probas.append(proba)
        
    print("DATASET SIZE: ", len(new_dataset['chosen']))
    avg = 0
    for arr in total_probas:
        for proba in arr:
            avg += proba
            
            
    avg = avg/(n*len(new_dataset))
    print("AVERAGE : ", avg)
    print(('\n'))
    if avg <= arguments.fallacy_threshold:
        print("THRESHOLD Validated!! AVG:", avg)
        return avg, args.output_dir + f'/iteration{iteration_number-1}'

        
    training_data = Dataset.from_dict(pd.DataFrame(new_dataset))
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=model,
        beta=args.beta,
        args=utils.get_training_args(args),
        train_dataset=training_data,
        eval_dataset=utils.get_data(args.data_dir, split='dev', return_type='dataset'),
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
    return avg, new_output_dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpo-model-path', required=True)
    parser.add_argument('--clf-path', default='models/fallacy_clf/howey_electra-base-mnli')
    parser.add_argument('--fallacy-threshold', default=0.13, type=float)
    
    return parser.parse_args()

if __name__ == '__main__':
    arguments = parse_args()
    main()