import json
from trl import CPOTrainer, CPOConfig
import argparse
import pandas as pd 
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import sys
sys.path.append('src/')
from datasets import load_dataset
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data-dir', default='data/dpo/')
    parser.add_argument('--model-name', default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--n-epochs', default=3, type=int)
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
    parser.add_argument('--max-length', default=100, type=int)
    parser.add_argument('--use-peft', default='false')
    parser.add_argument('--peft-config-r', default=16, type=int)
    parser.add_argument('--peft-config-lora-alpha', default=48, type=int)
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
        'prompt': '<s> [INST] ### Prompt: ' + put_capital_stance(example['prompt']) + f" [/INST]\n### Argument:" ,
        'chosen': example['chosen'] + ' </s>',
        'rejected': example['rejected'] + ' </s>',
    }
    
def main():
    args = parse_args()
    if args.data_dir[-1] != '/':
        args.data_dir += '/'
    
    input_dir=args.data_dir + '/'

    train_data = load_dataset('json', data_files=input_dir + 'train.json', split='train') 

    train_data = train_data.map(lambda sample: map_data(sample))
    model_name = args.model_name

    args.output_dir = f'{args.output_dir}/cpo_{model_name.split("/")[-1]}'
        

    if 't5' in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto')
        is_encoder_decoder = True
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        is_encoder_decoder = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token=tokenizer.unk_token

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM if is_encoder_decoder else TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        r=args.peft_config_r, 
        lora_alpha=args.peft_config_lora_alpha, 
        lora_dropout=args.peft_config_lora_dropout
        )
    
    model = PeftModel.from_pretrained(model, is_trainable=True, config=peft_config)
    model.print_trainable_parameters()  
    
    cpo_config = CPOConfig(
            output_dir=args.output_dir,               
            overwrite_output_dir=False,                  
            num_train_epochs=args.n_epochs,                   
            per_device_train_batch_size=args.batch_size,         
            learning_rate=args.learning_rate,                      
            warmup_steps=args.warmup_steps,                           
            weight_decay=args.weight_decay,                         
            adam_epsilon=args.adam_epsilon,                         
            save_steps=args.save_steps,            
            eval_steps=5000,          
            evaluation_strategy='steps', 
            logging_steps=10,                      
            save_total_limit=1,                         
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            beta=args.beta,
            max_prompt_length=args.max_length,
            max_length=args.max_length,
            is_encoder_decoder=is_encoder_decoder,
            generate_during_eval=False,
    )
    cpo_trainer = CPOTrainer(
        model=model,
        args=cpo_config,
        train_dataset=train_data,
        tokenizer=tokenizer,
    )
    
    cpo_trainer.train()
    print("SAVING MODEL at ", args.output_dir)
    with open(args.output_dir + '/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    cpo_trainer.save_model(args.output_dir)

   
if __name__ == '__main__':
    main()
