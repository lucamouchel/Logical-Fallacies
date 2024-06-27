import pandas as pd
import json
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from datasets import Dataset
import transformers
from transformers import TrainingArguments
import argparse
import torch
import sys
sys.path.append('src/')
from DPO.utils import get_training_args
from peft import LoraConfig, TaskType, get_peft_model
from datetime import datetime
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import RagRetriever, RagSequenceForGeneration, RagTokenForGeneration, RagTokenizer, pipeline
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

def formatting_prompts_func(example, task='argument'):
    ## Task = argument or claim
    data = []
    for i in tqdm(range(len(example['prompt'])), total=len(example['prompt'])):
        print(i)
        prompt = example['prompt'][i]
        context = example['context'][i]        
        completion = example['argument'][i]
        text = f"<s> [INST] ### Prompt: {prompt} \n### Context: {context} [/INST] \n### {task.title()}: {completion} </s>"
        data.append(text)     
    return data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/sft/')
    parser.add_argument('--model-name', required=True)
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
    parser.add_argument('--max-length', default=256, type=int)
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

    if "/" in model_name:
        output_directory =f'{args.output_dir}/sft_rag_{model_name.split("/")[-1]}_trl_{datetime.now()}'
    else: 
        output_directory =f'{args.output_dir}/sft_rag_{model_name}_trl_{datetime.now()}'
    args.output_dir = output_directory.replace(' ', '_')
    
    if 't5' in args.model_name.lower(): 
        model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, device_map='auto')
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)    
        
    tokenizer.pad_token=tokenizer.unk_token
    
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.peft_config_r, lora_alpha=args.peft_config_lora_alpha, lora_dropout=args.peft_config_lora_dropout)
    model = PeftModel.from_pretrained(model, is_trainable=True, config=peft_config)

    training_args = get_training_args(args)

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
