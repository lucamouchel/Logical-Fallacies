from datasets import load_dataset
import json
from operator import is_
from trl import KTOTrainer, KTOConfig
import transformers
from datasets import Dataset
import argparse
import pandas as pd 
from datetime import datetime
from datasets import load_dataset
import sys
sys.path.append('src/')
from peft import AutoPeftModelForCausalLM
from src.utils import get_training_args

def get_data():
    train_data = load_dataset('json', data_files="data/dpo/arguments/" + 'train.json', split='train') 
    converted_data = []
    seen_completions = []
    for entry in train_data:
        prompt = entry["prompt"]
        chosen_completion = entry["chosen"]
        rejected_completion = entry["rejected"]
        
        chosen_entry = {"prompt": prompt, "completion": chosen_completion, "label": True}
        rejected_entry = {"prompt": prompt, "completion": rejected_completion, "label": False}
        
        if chosen_completion not in seen_completions:
            converted_data.append(chosen_entry)
            seen_completions.append(chosen_completion)

        converted_data.append(rejected_entry)
    return Dataset.from_dict(pd.DataFrame(converted_data))



def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data-dir', default='data/dpo/')
    parser.add_argument('--ref-model-path', required=True)
    parser.add_argument('--desirable-weight', default=1.0, type=float)
    parser.add_argument('--undesirable-weight', default=1.0, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--n-epochs', default=10, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--eval-batch-size', default=32, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=2, type=int)
    parser.add_argument('--learning-rate', default=2e-4, type=float)
    parser.add_argument('--warmup-steps', default=100, type=int)
    parser.add_argument('--weight-decay', default=0.05, type=float)
    parser.add_argument('--adam-epsilon', default=1e-8, type=float)
    parser.add_argument('--save-steps', default=500, type=int)
    parser.add_argument('--logging-steps', default=300, type=int)
    parser.add_argument('--output-dir', default='models')
    parser.add_argument('--max-length', default=512, type=int)
    return parser.parse_args()

def put_capital_stance(prompt):
    if 'supporting argument' in prompt:
        return prompt.replace('supporting argument', 'SUPPORTING argument')
    elif 'counter argument' in prompt:
        return prompt.replace('counter argument', 'COUNTER argument')
    else:
        return prompt    
 
def map_data(example):
    return {
        'prompt': '<s> [INST] ### Prompt: ' + put_capital_stance(example['prompt']) + f" [/INST]\n### Argument:" ,
        'chosen': example['chosen'] + ' </s>',
        'rejected': example['rejected'] + ' </s>'
    }
    

def main():
    args = parse_args()
    if args.data_dir[-1] != '/':
        args.data_dir += '/'

    train_data = get_data()
    model_name = args.ref_model_path.split('/')[-1].split("_")[-1].lower()
    ref_model_path = args.ref_model_path    
    if '/' in model_name:
        output_directory =f'{args.output_dir}/kto_{model_name.split("/")[-1]}_{datetime.now()}'
    else: 
        output_directory =f'models/kto_{model_name}_{datetime.now()}'
        
    args.output_dir = output_directory.replace(' ', '_')
    training_args = get_training_args(args)
    is_encoder_decoder='t5' in model_name.lower()

    if 't5' in model_name.lower():
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(ref_model_path)
        ref_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(ref_model_path)
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(ref_model_path, is_trainable=True, device_map='auto')
        ref_model = AutoPeftModelForCausalLM.from_pretrained(ref_model_path, is_trainable=True, device_map='auto')

    tokenizer = transformers.AutoTokenizer.from_pretrained(ref_model_path)

   
    training_args = get_training_args(args)
    training_args = KTOConfig(
        beta=args.beta,
        desirable_weight=args.desirable_weight,
        undesirable_weight=args.undesirable_weight,
        max_prompt_length=args.max_length,
        max_length=args.max_length,
        is_encoder_decoder=is_encoder_decoder,
        **training_args.to_dict()
    )
    model.enable_input_require_grads()
    kto_trainer = KTOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer,
        )
    
    kto_trainer.train()
    print("SAVING MODEL at ", args.output_dir)
    
    with open(args.output_dir + '/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    kto_trainer.save_model('models/arguments/kto_mistral')

   
if __name__ == '__main__':
    main()


