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
from peft import LoraConfig, TaskType, get_peft_model
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
### python src/DPO/sft_with_trl.py --model-name=google/flan-t5-base --data-dir=data/dpo/claims --n-epochs=2 --batch-size=8 --gradient-accumulation-steps=2 --output-dir=models/test/sft_flan-t5-base_trl --is-encoder-decoder=true 
"""
def get_data(data_dir, split='train'):
    if data_dir[-1] != '/': 
        data_dir += '/'
    with open(f'{data_dir+split}.json', 'r') as f:
        df = json.load(f)
    
    df = pd.DataFrame(df).drop(columns=['rejected'])
    df.drop_duplicates(subset=['chosen'], inplace=True)
    return Dataset.from_dict(df)
"""
def formatting_prompts_func(example):
    data = []
    for i in range(len(example['prompt'])):
        prompt = example['prompt'][i]
        if 'supporting argument' in prompt:
            prompt = prompt.replace('supporting argument', 'SUPPORTING argument')
        else:
            prompt = prompt.replace('counter argument', 'COUNTER argument')
            
        completion = example['argument'][i]
        text = f"<s> [INST] ### Prompt: {prompt} [/INST] \n### Argument: {completion} </s>"
        data.append(text)
        
    return data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/dpo/arguments/')
    parser.add_argument('--model-name', default='google/gemma-2b')
    parser.add_argument('--n-epochs', default=10, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--eval-steps', default=80, type=int)
    parser.add_argument('--eval-batch-size', default=32, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=2, type=int)
    parser.add_argument('--learning-rate', default=5e-4, type=float)
    parser.add_argument('--warmup-steps', default=0, type=int)
    parser.add_argument('--weight-decay', default=0.01, type=float)
    parser.add_argument('--adam-epsilon', default=1e-8, type=float)
    parser.add_argument('--save-steps', default=80, type=int)
    parser.add_argument('--logging-steps', default=80, type=int)
    parser.add_argument('--output-dir', default='models')
    parser.add_argument('--max-length', default=128, type=int)
    parser.add_argument('--use-peft', default='false')
    parser.add_argument('--peft-config-r', default=16, type=int)
    parser.add_argument('--peft-config-lora-alpha', default=32, type=int)
    parser.add_argument('--peft-config-lora-dropout', default=0.05, type=float)
    return parser.parse_args()
#python src/DPO/sft.py --model-name=google/gemma-2b --batch-size=16 --use-peft=false

def main():
    args = parse_args()
    
    model_name = args.model_name
    
    train_data = load_dataset('json', data_files='data/sft_refiner/train.json', split='train')
 
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
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    #### IS THIS THE ONLY ADDITION TO THE TOKENIZER we had in the end??
    tokenizer.pad_token=tokenizer.unk_token
    
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05)
    training_args = get_training_args(args)

    model = get_peft_model(model, peft_config)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        #eval_dataset=val_data,
        formatting_func=formatting_prompts_func,
        max_seq_length=args.max_length,
        ##### DID we have anything extra here? max-length?
        #peft_config=peft_config if args.use_peft == 'true' else None,
        )
    
    trainer.train()
    print("SAVING MODEL at ", args.output_dir)
    trainer.save_model(args.output_dir)
    
if __name__ == '__main__':
    main()
