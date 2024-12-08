from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import argparse
from datetime import datetime
from peft import LoraConfig, TaskType
import wandb
from tqdm import tqdm
import torch
import pathlib
import json 

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
    prompt = '<s> [INST] ### Prompt: ' + sample['prompt'] + f" [/INST]\n### Argument:"
    sample['query'] = prompt
    return sample


def train(args):
    train_data = load_dataset('json', data_files='data/sft/arguments/train.json', split='train')
    model_name = args.ref_model_path.split('sft_')[-1]

    args.output_dir = f'models/ppo_{model_name}'
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(args.output_dir + '/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    config = PPOConfig(
        learning_rate=args.learning_rate, 
        batch_size=args.batch_size, 
        ppo_epochs=args.n_epochs, 
        mini_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        reward_model=args.reward_model_path,
        model_name=model_name
        )
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.ref_model_path).to('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(args.ref_model_path, padding_side='left')
    
    train_data = train_data.map(map_data)

    reward_model_path = args.reward_model_path
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path).to('cuda:1')
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    
    trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer, dataset=train_data)
    
    
    generation_kwargs = {
        "do_sample": True,
        "max_new_tokens": 30,
        "no_repeat_ngram_size": 2,
    }

    for _ in tqdm(range(trainer.config.ppo_epochs)):
        for j, batch in tqdm(enumerate(trainer.dataloader)):
            queries = batch['query']
            
            tokenized = tokenizer(queries, padding='max_length', max_length=128, truncation=True)

            input_ids = [torch.tensor(ids).to('cuda:0') for ids in tokenized['input_ids']]
            responses = [trainer.generate(ids, return_prompt=False, **generation_kwargs, pad_token_id=tokenizer.pad_token_id)[0].to('cuda:0') for ids in input_ids]
            batch['response'] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in responses]
        
            tokens = reward_tokenizer(batch['response'], padding=True, truncation=True)
            tokens = {k: torch.tensor(v).to('cuda:1') for k, v in tokens.items()}
            rewards = reward_model(**tokens)
            rewards = rewards.logits[:, 0]
            rewards = [torch.tensor(r).to('cuda:0') for r in rewards]
            
            if j % 100 == 0 and j != 0:
                for i, (response, reward)in enumerate(zip(batch['response'], rewards)):
                    print("TOPIC: ", batch['query'][i].split('topic: ')[-1])
                    print("STANCE: ", 'SUPPORTING' if 'SUPPORTING' in batch['query'][i] else 'COUNTER')
                    print(f"Reward: {reward:.3f} \t----\t Response: {response}\n\n")
                    
            stats = trainer.step(input_ids, responses, rewards)
            trainer.log_stats(stats, batch, rewards)
            
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
