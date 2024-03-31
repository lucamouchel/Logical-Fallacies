from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import argparse
from datetime import datetime
from peft import LoraConfig, TaskType
import wandb
from tqdm import tqdm
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/dpo/')
    parser.add_argument('--model-name', default='models/Llama-2-7b-hf')
    parser.add_argument('--ref-model-path', default='models/sft_flan-t5-large_trl')
    parser.add_argument('--reward-model-path', required=True)
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
    parser.add_argument('--max-length', default=128, type=int)
    parser.add_argument('--use-peft', default='false')
    parser.add_argument('--peft-config-r', default=16, type=int)
    parser.add_argument('--peft-config-lora-alpha', default=32, type=int)
    parser.add_argument('--peft-config-lora-dropout', default=0.05, type=float)
    return parser.parse_args()

def get_training_args(args):
    return TrainingArguments(
            output_dir=args.output_dir,               
            overwrite_output_dir=False,                  
            num_train_epochs=args.n_epochs,                   
            per_device_train_batch_size=args.batch_size,         
            learning_rate=args.learning_rate,                      
            warmup_steps=args.warmup_steps,                           
            weight_decay=args.weight_decay,                         
            adam_epsilon=args.adam_epsilon,                         
            save_steps=args.save_steps,                       
            logging_steps=args.logging_steps,                      
            save_total_limit=2,                         
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )       
    
def map_data(sample):
    ## sample has prompt + argument -- we use the sft training prompts
    prompt = '<s> [INST] ### Prompt: ' + sample['prompt'] + f" [/INST]\n### Argument:"
    sample['query'] = prompt
    return sample


def main():
    wandb.init(mode="disabled")
    args = parse_args()
    train_data = load_dataset('json', data_files='data/sft/arguments/train.json', split='train')
    
    #### PPO expects only the prompts and then uses the sft model and reward model to do stuff
    model_name = args.model_name
    training_args = get_training_args(args)
    
    config = PPOConfig(
        learning_rate=args.learning_rate, 
        batch_size=args.batch_size, 
        ppo_epochs=args.n_epochs, 
        mini_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        reward_model=args.reward_model_path,
        model_name=model_name
        )
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.ref_model_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.ref_model_path, padding_side='left')
    
    train_data = train_data.map(map_data)

    reward_model_path = args.reward_model_path
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path).to('cuda')
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    
    trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer, dataset=train_data)
    
    generation_kwargs = {
        "do_sample": True,
        "max_new_tokens": 30,
        "no_repeat_ngram_size": 2,
    }
    
    for epoch in tqdm(range(trainer.config.ppo_epochs)):
        for batch in tqdm(trainer.dataloader):
            queries = batch['query']
            tokenized = tokenizer(queries, padding='max_length', max_length=128, truncation=True)

            input_ids = [torch.tensor(ids).to('cuda') for ids in tokenized['input_ids']]
            responses = [trainer.generate(ids, return_prompt=False, **generation_kwargs) for ids in input_ids]
            batch['response'] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in responses]
            print(batch['response'])
                
            
            #texts = [f"{p} {r}" for p, r in zip(batch['query'], batch['response'])]
            #print(texts)
            tokens = reward_tokenizer(batch['response'], padding=True, truncation=True)
            tokens = {k: torch.tensor(v).to('cuda') for k, v in tokens.items()}
            rewards = reward_model(**tokens)
            rewards = rewards.logits[:, 0]
            rewards = [torch.tensor(r) for r in rewards]
            print(rewards)
            
            print()
            print()
            print([ids for ids in input_ids])
            ids = [ids for ids in input_ids]
            
            a = [torch.cat([q, r]) for q, r in zip(input_ids, responses)]
            stats = trainer.step([ids for ids in input_ids], responses, rewards)
            trainer.log_stats(stats, batch, rewards)
            
    
    
    
    if '/' in model_name:
        output_directory =f'models/arguments/ppo_{model_name.split("/")[-1]}_{datetime.now()}'
    else:
        output_directory =f'models/arguments/ppo_{model_name}_{datetime.now()}'

    args.output_dir = output_directory.replace(' ', '_')
    trainer.save_model(args.output_dir)
if __name__ == '__main__':
    main()