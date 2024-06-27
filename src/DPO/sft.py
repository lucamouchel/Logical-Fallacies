import pandas as pd
import json
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from datasets import Dataset
import transformers
from transformers import TrainingArguments
import argparse
import torch
from utils import get_training_args
from peft import LoraConfig, TaskType, PeftModel
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def formatting_prompts_func(example):
    ## Task = argument or claim
    data = []
    for i in range(len(example['prompt'])):
        prompt = example['prompt'][i]
        completion = example['argument'][i]
        text = f"<s> [INST] ### Prompt: {prompt} [/INST] \n### Argument: {completion} </s>"
        data.append(text)      
    return data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/sft/')
    parser.add_argument('--model-name', default='google/gemma-2b')
    parser.add_argument('--n-epochs', default=10, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--eval-steps', default=80, type=int)
    parser.add_argument('--eval-batch-size', default=32, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=2, type=int)
    parser.add_argument('--learning-rate', default=2e-4, type=float)
    parser.add_argument('--warmup-steps', default=0, type=int)
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--adam-epsilon', default=1e-8, type=float)
    parser.add_argument('--save-steps', default=80, type=int)
    parser.add_argument('--logging-steps', default=80, type=int)
    parser.add_argument('--output-dir', default='models')
    parser.add_argument('--max-length', default=512, type=int)
    parser.add_argument('--use-peft', default='false')
    parser.add_argument('--peft-config-r', default=16, type=int)
    parser.add_argument('--peft-config-lora-alpha', default=32, type=int)
    parser.add_argument('--peft-config-lora-dropout', default=0.05, type=float)
    return parser.parse_args()
#python src/DPO/sft.py --model-name=google/gemma-2b --batch-size=16 --use-peft=false

def main():
    args = parse_args()
    
    model_name = args.model_name
    if args.data_dir[-1] != '/':
        args.data_dir += '/'

    input_dir = args.data_dir + '/'
    train_data = load_dataset('json', data_files=input_dir + 'train.json', split='train')
 
    if "/" in model_name :
        output_directory =f'{args.output_dir}/sft_{model_name.split("/")[-1]}_trl_{datetime.now()}'
    else: 
        output_directory =f'{args.output_dir}/sft_{model_name}_trl_{datetime.now()}'
    args.output_dir = output_directory.replace(' ', '_')
    
    if 't5' in args.model_name.lower(): ### we use T5 but you can use some other model
        model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    elif 'llama' in args.model_name.lower():
        model = transformers.LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, device_map='auto')
        tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)    
    else: ## if we use Gemma we can just use the AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token=tokenizer.unk_token
    
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05)
    training_args = get_training_args(args)

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.peft_config_r, lora_alpha=args.peft_config_lora_alpha, lora_dropout=args.peft_config_lora_dropout)
    model = PeftModel.from_pretrained(model, is_trainable=True, config=peft_config)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        formatting_func=formatting_prompts_func,
        max_seq_length=args.max_length,
    )
    
    trainer.train()
    print("SAVING MODEL at ", args.output_dir)
    trainer.save_model(args.output_dir)
    
if __name__ == '__main__':
    main()
