import json
from operator import is_
from trl import DPOTrainer, DPOConfig
import transformers
from transformers import TrainingArguments
from datasets import Dataset
import argparse
import pandas as pd 
from datetime import datetime
import sys 
sys.path.append('src/')
from custom_dpo_trainer import CustomDPO

from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
#from utils import get_training_args

task='arguments'

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data-dir', default='data/dpo/')
    parser.add_argument('--ref-model-path', default='models/sft_flan-t5-large_trl')
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--n-epochs', default=10, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--eval-batch-size', default=32, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=2, type=int)
    parser.add_argument('--learning-rate', default=2e-5, type=float)
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
    parser.add_argument('--low-resource', action='store_true')
    return parser.parse_args()

## python src/DPO/dpo.py --ref-model-path=models/arguments/sft_llama --beta=0.3 --n-epochs=3 
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

    input_dir=args.data_dir + task + '/'

    train_data = load_dataset('json', data_files=input_dir + 'train.json', split='train') 
    train_data = train_data.map(lambda sample: map_data(sample, task=task))
    
    ref_model_path = args.ref_model_path    
    model_name = 'llama' if 'llama' in ref_model_path.lower() else "mistral"
    if '/' in model_name:
        output_directory =f'{args.output_dir}/{task}/dpo_{model_name.split("/")[-1]}_{datetime.now()}'
    else: 
        output_directory =f'models/{task}/dpo_{model_name}_{datetime.now()}'
    args.output_dir = output_directory.replace(' ', '_')
        
    model = AutoPeftModelForCausalLM.from_pretrained(ref_model_path, device_map='auto')
   # ref_model = AutoPeftModelForCausalLM.from_pretrained(ref_model_path, device_map='auto')
    tokenizer = transformers.AutoTokenizer.from_pretrained(ref_model_path)

    model.enable_input_require_grads()

    with open('data/dpo/arguments/test.json', 'r') as f:
        test_data = json.load(f)
        test_data = test_data[:30]
        test_data = Dataset.from_dict(pd.DataFrame(test_data))
        test_data = test_data.map(lambda sample: map_data(sample, task=task))

    training_args = DPOConfig(
            output_dir=args.output_dir,               
            overwrite_output_dir=False,                  
            num_train_epochs=args.n_epochs,                   
            per_device_train_batch_size=args.batch_size,         
            learning_rate=args.learning_rate,                      
            warmup_steps=args.warmup_steps,                           
            weight_decay=args.weight_decay,                         
            adam_epsilon=args.adam_epsilon,                         
            save_steps=args.save_steps,            
            eval_steps=400,          
            evaluation_strategy='steps', 
            logging_steps=10,                      
            save_total_limit=1,                         
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            beta=args.beta,
            max_prompt_length=150,
            max_length=args.max_length,
            is_encoder_decoder=False,
            generate_during_eval=True,
    )

    dpo_trainer = CustomDPO(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
    )

    dpo_trainer.train()
    print("SAVING MODEL at ", args.output_dir)
    
    with open(args.output_dir + '/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    dpo_trainer.save_model(args.output_dir)
    if isinstance(dpo_trainer, CustomDPO):
        dpo_trainer.fallacy_clf.save_pretrained(f'models/fallacy_clf/howey_electra-large-mnli_{datetime.now()}')

   
if __name__ == '__main__':
    main()

