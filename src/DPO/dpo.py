import json
from operator import is_
from trl import DPOTrainer
import transformers
from transformers import TrainingArguments
from datasets import Dataset
import argparse
import string 
import pandas as pd 
from peft import LoraConfig, TaskType, get_peft_model
from datetime import datetime
import torch
from tqdm import tqdm
import gc
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
import utils 


def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data-dir', default='data/dpo/arguments/')
    parser.add_argument('--model-name', default='google/flan-t5-base')
    parser.add_argument('--ref-model-path', default='models/sft_flan-t5-large_trl')
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--n-epochs', default=10, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--eval-batch-size', default=32, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=2, type=int)
    parser.add_argument('--learning-rate', default=5e-7, type=float)
    parser.add_argument('--warmup-steps', default=0, type=int)
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--adam-epsilon', default=1e-8, type=float)
    parser.add_argument('--save-steps', default=500, type=int)
    parser.add_argument('--logging-steps', default=300, type=int)
    parser.add_argument('--eval-steps', default=600, type=int)
    parser.add_argument('--output-dir', default='models')
    parser.add_argument('--max-length', default=512, type=int)
    parser.add_argument('--use-peft', default='false')
    parser.add_argument('--peft-config-r', default=16, type=int)
    parser.add_argument('--peft-config-lora-alpha', default=32, type=int)
    parser.add_argument('--peft-config-lora-dropout', default=0.05, type=float)
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
        'prompt': '<s> [INST] ### Prompt: ' + put_capital_stance(example['prompt']) + ' [/INST]\n### Argument' ,
        'chosen': example['chosen'] + ' </s>',
        'rejected': example['rejected'] + ' </s>'
    }
    

def main():
    args = parse_args()
        
    train_data = load_dataset('json', data_files='data/dpo/arguments_dpo/train.json', split='train')  
    train_data = train_data.map(map_data)
    
    model_name = args.model_name
    ref_model_path = args.ref_model_path    
    if '/' in model_name:
        output_directory =f'{args.output_dir}/dpo_{model_name.split("/")[-1]}_trl_{datetime.now()}'
    else: 
        output_directory =f'models/dpo_{model_name}_trl_{datetime.now()}'
        
    args.output_dir = output_directory.replace(' ', '_')
    training_args = utils.get_training_args(args)
    
    if 't5' in model_name.lower():
        is_encoder_decoder = True
        model = transformers.T5ForConditionalGeneration.from_pretrained(ref_model_path)
        ref_model = transformers.T5ForConditionalGeneration.from_pretrained(ref_model_path)
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    elif 'llama' in model_name.lower():
        is_encoder_decoder = False
        model = AutoPeftModelForCausalLM.from_pretrained(ref_model_path)
        #ref_model = transformers.LlamaForCausalLM.from_pretrained(ref_model_path)
        tokenizer = transformers.LlamaTokenizer.from_pretrained(ref_model_path)
    else:
        is_encoder_decoder = False
        model = transformers.AutoModelForCausalLM.from_pretrained(ref_model_path, device_map='auto')
        ref_model = transformers.AutoModelForCausalLM.from_pretrained(ref_model_path, device_map='auto')
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        
    task_type = TaskType.SEQ_2_SEQ_LM if is_encoder_decoder else TaskType.CAUSAL_LM 
    

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        beta=args.beta,
        args=training_args,
        train_dataset=train_data,
        is_encoder_decoder=is_encoder_decoder,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_target_length=args.max_length,
        #peft_config=peft_config if args.use_peft == 'true' else None,
        generate_during_eval=True)
    
    dpo_trainer.train()
    print("SAVING MODEL at ", args.output_dir)
    
    with open(args.output_dir + '/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    dpo_trainer.save_model(args.output_dir)

   
if __name__ == '__main__':
    main()

