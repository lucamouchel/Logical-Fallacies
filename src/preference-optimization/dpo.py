import json
from trl import DPOTrainer, DPOConfig
import transformers
import argparse
from datetime import datetime
import sys 
sys.path.append('src/')
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM


def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data-dir', default='data/dpo/')
    parser.add_argument('--ref-model-path', default='models/sft_llama')
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--n-epochs', default=4, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--eval-batch-size', default=32, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=2, type=int)
    parser.add_argument('--learning-rate', default=1e-5, type=float)
    parser.add_argument('--warmup-steps', default=100, type=int)
    parser.add_argument('--weight-decay', default=0.05, type=float)
    parser.add_argument('--adam-epsilon', default=1e-8, type=float)
    parser.add_argument('--save-steps', default=500, type=int)
    parser.add_argument('--logging-steps', default=300, type=int)
    parser.add_argument('--output-dir', default='models')
    parser.add_argument('--max-length', default=128, type=int)
    return parser.parse_args()

def map_data(example):
    prompt = example['prompt']
    if 'supporting argument' in prompt:
        prompt = prompt.replace('supporting argument', 'SUPPORTING argument')
    elif 'counter argument' in prompt:
        prompt = prompt.replace('counter argument', 'COUNTER argument')

    return {
        'prompt': '<s> [INST] ### Prompt: ' + prompt + f" [/INST]\n### Argument: " ,
        'chosen': example['chosen'] + ' </s>',
        'rejected': example['rejected'] +  " </s>",
        'fallacy_type': example['fallacy_type']
    }
    
def main():
    args = parse_args()
    if args.data_dir[-1] != '/':
        args.data_dir += '/'

    input_dir=args.data_dir  + '/'

    train_data = load_dataset('json', data_files=input_dir + 'train.json', split='train') 
    train_data = train_data.map(map_data)
    
    ref_model_path = args.ref_model_path    
    model_name = ref_model_path.split('sft_')[-1]
    args.output_dir = f'{args.output_dir}/dpo_{model_name}'
        
    if 't5' in ref_model_path.lower():
        model = AutoPeftModelForSeq2SeqLM.from_pretrained(ref_model_path, is_trainable=True, device_map='auto')
        is_encoder_decoder = True
    else:
       model = AutoPeftModelForCausalLM.from_pretrained(ref_model_path, is_trainable=True, device_map='auto')
       is_encoder_decoder = False   

    tokenizer = transformers.AutoTokenizer.from_pretrained(ref_model_path)

    model.print_trainable_parameters()

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
            eval_steps=4000,          
            eval_strategy='steps', 
            logging_steps=10,                      
            save_total_limit=1,                         
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            beta=args.beta,
            max_prompt_length=args.max_length,
            max_length=args.max_length,
            is_encoder_decoder=is_encoder_decoder,
            generate_during_eval=True,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer,
    )   

    dpo_trainer.train()
    print("SAVING MODEL at ", args.output_dir)
    
    with open(args.output_dir + '/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    dpo_trainer.save_model(args.output_dir)
    
if __name__ == '__main__':
    main()

