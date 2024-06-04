
import json
import argparse
import transformers
import torch
import gc 
import pandas as pd 
from peft import AutoPeftModelForCausalLM
from datasets import Dataset
from tqdm import tqdm
from trl import DPOTrainer, ORPOConfig, ORPOTrainer, DPOConfig
import utils
from collections import namedtuple

CLASSES = {'Not a Fallacy': 0,  'faulty generalization': 1, 'false causality': 2, 'fallacy of relevance': 3, 'fallacy of extension': 4, 'equivocation': 5, 'ad populum': 6, 'appeal to emotion': 7, 'ad hominem': 8, 'circular reasoning': 9, 'fallacy of credibility': 10, 'fallacy of logic': 11, 'false dilemma': 12, 'intentional': 13}
INVERSE = {v: k for k, v in CLASSES.items()}
new_classes= {'Not a Fallacy': 0,
 'ad hominem': 1,
 'ad populum': 2,
 'appeal to emotion': 3,
 'circular reasoning': 4,
 'equivocation': 5,
 'fallacy of relevance': 6,
 'false causality': 7,
 'false dilemma': 8,
 'faulty generalization': 9}

def main():
    model_path = arguments.model_path
    if model_path[-1] != '/':
        model_path += '/'

    args = dict(json.load(open(model_path+'args.json', 'r')))
    args = namedtuple('DotDict', args.keys())(**args)    
    
    model = AutoPeftModelForCausalLM.from_pretrained(model_path, is_trainable=True, device_map='auto') 
    model.print_trainable_parameters()
    ref_model = AutoPeftModelForCausalLM.from_pretrained(model_path, is_trainable=False, device_map='auto')
    ref_model.print_trainable_parameters()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    fallacy_classifier = transformers.AutoModelForSequenceClassification.from_pretrained(arguments.clf_path)
    clf_tokenizer = transformers.AutoTokenizer.from_pretrained(arguments.clf_path)

    iteration = 1
    while True:
        if iteration > 1:
            ref_model = AutoPeftModelForCausalLM.from_pretrained(new_output_dir, is_trainable=False, device_map='auto')
        avg, new_output_dir = run_self_reward(
            args=args, 
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            clf=fallacy_classifier, 
            clf_tokenizer=clf_tokenizer, 
            iteration_number=iteration
        )
        print("ITERATION ", iteration, " AVERAGE : ", avg)
        iteration += 1
        del ref_model 
        if avg <= arguments.fallacy_threshold:
            break

    print("FINAL MODEL saved at ", new_output_dir)

def map_data(example):
    return {
        'prompt': '<s> [INST] ### Prompt: ' + example['prompt'] + f" [/INST]\n### Argument:" ,
        'chosen': example['chosen'] + ' </s>',
        'rejected': example['rejected'] + f" </s>"
    }
    
    
def run_self_reward(args, model, ref_model, clf, clf_tokenizer, tokenizer, iteration_number, is_encoder_decoder=False):
    print("Running self reward loop -- iteration", iteration_number)
    clf.eval()
    def fallacy_proba(instruction, ys):
        if instruction:
            inputs = [instruction + ' ' + y for y in ys]
        else:
            inputs = ys
        inputs = clf_tokenizer(inputs, add_special_tokens=True, return_tensors='pt', padding=True, truncation=True, max_length=80).to(clf.device)
        with torch.no_grad():
            return torch.softmax(clf(**inputs).logits, dim=-1)
    
    GENERATION_KWARGS = {'max_new_tokens': 30, 'no_repeat_ngram_size': 2, 'do_sample': True, 'top_p': 0.75, 'temperature': 0.8}    
    def generate(prompt: str, model, tokenizer, n):
        """Main function for text generation."""
        tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=128, truncation=True).to(model.device)
        with torch.no_grad():
            output = model.generate(**tokenized_prompt,
                                    **GENERATION_KWARGS,
                                    num_return_sequences=n,
                                    pad_token_id=tokenizer.eos_token_id)
        return tokenizer.batch_decode(output, skip_special_tokens=True)

    with open('data/dpo/arguments/dpo_with_fallacies_test.json', 'r') as f:
        test = json.load(f)

    n = 8 ## generate 4 responses for each prompt
    test = test[:1200:4]

    new_dataset = []    
    total_fallacy_proba = 0
    print("Sampling responses and generating fallacy probabilities.")
    total_rejected = 0
    for i, sample in tqdm(enumerate(test), total=len(test)):
        instruction = sample['prompt']
        if 'supporting' in instruction:
            new_instrctution = instruction.replace('supporting', 'SUPPORTING')
        elif 'counter' in prompt:
            new_instrctution = instruction.replace('counter', 'COUNTER')
            
        prompt = '<s> [INST] ### Prompt: ' + new_instrctution + " [/INST]\n### Argument: "
        ys = generate(prompt, model, tokenizer, n) 
        temp_dataset = []
        ys = list(map(lambda y: y.split('### Argument: ')[-1].strip(), ys))
        print(ys)
        probas = fallacy_proba(instruction, ys) 
        predicted_fallacies = [torch.argmax(proba).item() for proba in probas]
        print(predicted_fallacies)
        probas = [torch.max(proba).item() for proba in probas]

        temp_dataset = list(zip(ys, probas, predicted_fallacies))        

        chosen_samples= [(y, probas) for y, probas, fallacy in temp_dataset if fallacy == 0]
        rejected_samples = [(y, probas, fallacy) for y, probas, fallacy in temp_dataset if fallacy != 0]
        total_rejected += len(rejected_samples)
        rejected_probas = [proba for _, proba, _ in rejected_samples]
        total_fallacy_proba += sum(rejected_probas)
        print('REJECTED SAMPLEs')
        print(rejected_samples)
        if i % 30 == 0 and i > 0:
            print(len(new_dataset))
            print("Total Fallacy samples: ", total_rejected)
        for y_chosen, y_chosen_proba in chosen_samples:
            for y_rejected, y_rejected_proba, y_fallacy in rejected_samples:
                
                if y_chosen_proba >= 0.5 and y_rejected_proba >= 0.5:
                    new_dataset.append({
                        'prompt': instruction,
                        'chosen': y_chosen,
                        'rejected': y_rejected,
                        'fallacy_type': y_fallacy
                    })
                

    print(new_dataset)
    print("DATASET SIZE: ", len(new_dataset))
    # avg = sum(total_probas)
    # avg = avg/(n*len(new_dataset))
    avg = total_rejected / (n*len(test))
    print("TOTAL AMOUNT OF Fallacies: ", total_rejected)

    if avg <= arguments.fallacy_threshold:
        print("THRESHOLD Validated!! AVG:", avg)
        return avg, args.output_dir + f'/iteration{iteration_number-1}'
 
    training_data = Dataset.from_dict(pd.DataFrame(new_dataset))
    training_data.map(map_data)
    training_args = utils.get_training_args(args)
    
    optimization_algorithm = arguments.optim_algorithm
    
    if optimization_algorithm == 'DPO':
        config = DPOConfig(
            **training_args.to_dict(),
            max_length=args.max_length,
            max_prompt_length=args.max_length,
            max_target_length=args.max_length,
            is_encoder_decoder=is_encoder_decoder,


        )
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            beta=args.beta,
            args=config,
            train_dataset=training_data,
            tokenizer=tokenizer,
        )

        dpo_trainer.train()
        new_output_dir = args.output_dir + f'/iteration{iteration_number}'
        dpo_trainer.save_model(new_output_dir)
        
        
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
    parser.add_argument('--model-path', default='models/arguments/sft_llama')
    parser.add_argument('--clf-path', default='models/fallacy/clf')
    parser.add_argument('--fallacy-threshold', default=0.05, type=float)
    parser.add_argument('--optim-algorithm', default='DPO', help='DPO, ORPO')
    return parser.parse_args()

if __name__ == '__main__':
    arguments = parse_args()
    main()