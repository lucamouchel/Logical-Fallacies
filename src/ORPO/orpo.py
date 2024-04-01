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
from transformers import AutoModelForCausalLM
import DPO.utils
from trl import ORPOConfig, ORPOTrainer
from DPO.utils import get_training_args


def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data-dir', default='data/dpo/')
    parser.add_argument('--task', default='arguments') ## arguments or claims
    parser.add_argument('--model-name', default='google/flan-t5-base')
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
    parser.add_argument('--max-length', default=128, type=int)
    parser.add_argument('--use-peft', default='false')
    parser.add_argument('--peft-config-r', default=16, type=int)
    parser.add_argument('--peft-config-lora-alpha', default=32, type=int)
    parser.add_argument('--peft-config-lora-dropout', default=0.05, type=float)
    return parser.parse_args()


def map_data(example, task):
    return {
        'prompt': '<s> [INST] ### Prompt: ' + example['prompt'] + f" [/INST]\n### Argument:" ,
        'chosen': example['chosen'] + ' </s>',
        'rejected': example['rejected'] + ' </s>'
    }
    

def main():
    args = parse_args()
    if args.data_dir[-1] != '/':
        args.data_dir += '/'

    input_dir=args.data_dir + args.task + '/'

    train_data = load_dataset('json', data_files=input_dir + 'train.json', split='train')  
    train_data = train_data.map(lambda sample: map_data(sample, task=args.task))
    model_name = args.model_name

    if '/' in model_name:
        output_directory =f'{args.output_dir}/{args.task}/orpo_{model_name.split("/")[-1]}_{datetime.now()}'
    else: 
        output_directory =f'models/{args.task}/orpo_{model_name}_{datetime.now()}'
        
    args.output_dir = output_directory.replace(' ', '_')
    training_args = get_training_args(args)

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token=tokenizer.unk_token

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05)
    training_args = get_training_args(args)
    model = get_peft_model(model, peft_config)

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
        train_dataset=train_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    orpo_trainer.train()
    print("SAVING MODEL at ", args.output_dir)
    
    with open(args.output_dir + '/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    orpo_trainer.save_model(args.output_dir)

   
if __name__ == '__main__':
    main()

