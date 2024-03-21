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
from utils import get_training_args


def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data-dir', default='data/dpo/')
    parser.add_argument('--task', required=True) ## arguments or claims
    parser.add_argument('--model-name', default='google/flan-t5-base')
    parser.add_argument('--ref-model-path', default='models/sft_flan-t5-large_trl')
    parser.add_argument('--beta', default=0.5, type=float)
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
    parser.add_argument('--use-peft', default='false')
    parser.add_argument('--peft-config-r', default=16, type=int)
    parser.add_argument('--peft-config-lora-alpha', default=32, type=int)
    parser.add_argument('--peft-config-lora-dropout', default=0.05, type=float)
    return parser.parse_args()

def put_capital_stance(prompt, task):
    if task == 'claims': ## claims data is already mapped to capital stance
        return prompt 

    if 'supporting argument' in prompt:
        return prompt.replace('supporting argument', 'SUPPORTING argument')
    elif 'counter argument' in prompt:
        return prompt.replace('counter argument', 'COUNTER argument')
    else:
        return prompt    
 
def map_data(example, task):
    return {
        'prompt': '<s> [INST] ### Prompt: ' + put_capital_stance(example['prompt'], task=task) + f" [/INST]\n### {'Argument' if task=='arguments' else 'Claim'}:" ,
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
    ref_model_path = args.ref_model_path    
    if '/' in model_name:
        output_directory =f'{args.output_dir}/{args.task}/dpo_{model_name.split("/")[-1]}_{datetime.now()}'
    else: 
        output_directory =f'models/{task}/dpo_{model_name}_{datetime.now()}'
        
    args.output_dir = output_directory.replace(' ', '_')
    training_args = utils.get_training_args(args)
    is_encoder_decoder='t5' in model_name.lower()

    if 't5' in model_name.lower():
        model = transformers.T5ForConditionalGeneration.from_pretrained(ref_model_path)
        ref_model = transformers.T5ForConditionalGeneration.from_pretrained(ref_model_path)
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    elif 'llama' in model_name.lower():
        model = AutoPeftModelForCausalLM.from_pretrained(ref_model_path, device_map='auto')
        ref_model = AutoPeftModelForCausalLM.from_pretrained(ref_model_path, device_map='auto')
        tokenizer = transformers.LlamaTokenizer.from_pretrained(ref_model_path)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(ref_model_path, device_map='auto')
        ref_model = transformers.AutoModelForCausalLM.from_pretrained(ref_model_path, device_map='auto')
        tokenizer = transformers.AutoTokenizer.from_pretrained(ref_model_path)

    task_type = TaskType.CAUSAL_LM if is_encoder_decoder else TaskType.CAUSAL_LM 
#    model.config.use_cache = False 
    #model.requires_grad = Tre
#    tokenizer.pad_token=tokenizer.unk_token

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05)
    training_args = get_training_args(args)
    #model = get_peft_model(model, peft_config)

    model.enable_input_require_grads()
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=args.beta,
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer,
        max_prompt_length=150,
        max_length=args.max_length,
        is_encoder_decoder=is_encoder_decoder,
        #peft_config=peft_config if args.use_peft == 'true' else None,
        )
    
    dpo_trainer.train()
    print("SAVING MODEL at ", args.output_dir)
    
    with open(args.output_dir + '/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    dpo_trainer.save_model(args.output_dir)

   
if __name__ == '__main__':
    main()

