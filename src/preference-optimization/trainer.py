from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
from FIPO.FIPOTrainer import FIPOTrainer
from FIPO.FIPOConfig import FIPOConfig
from datasets import load_dataset
import torch.nn as nn
from tqdm import tqdm 
import argparse
import pathlib
import torch
import sys 
sys.path.append('src/')
from trl import (
    SFTConfig, 
    DPOConfig, 
    CPOConfig, 
    KTOConfig, 
    PPOConfig, 
    SFTTrainer, 
    DPOTrainer, 
    CPOTrainer, 
    KTOTrainer, 
    PPOTrainer, 
    AutoModelForCausalLMWithValueHead, 
    AutoModelForSeq2SeqLMWithValueHead
)
from peft import (
    AutoPeftModelForCausalLM, 
    AutoPeftModelForSeq2SeqLM,
    LoraConfig, 
    TaskType, 
    get_peft_model
)

CLASSES = {'Not a Fallacy': 0,  'faulty generalization': 1, 'false causality': 2, 'fallacy of relevance': 3, 'fallacy of extension': 4, 'equivocation': 5, 'ad populum': 6, 'appeal to emotion': 7, 'ad hominem': 8, 'circular reasoning': 9, 'fallacy of credibility': 10, 'fallacy of logic': 11, 'false dilemma': 12, 'intentional': 13}
INVERSE = {v: k for k, v in CLASSES.items()}


def parse_args():
    parser = argparse.ArgumentParser() 

    parser.add_argument('--train-using', type=str, required=True, help="training to perform -- either sft or preference optimization (sft, dpo, cpo, kto, fipo, ppo)")
    parser.add_argument('--train-data', type=str, default='data/preference-data/train.json', help='path to json training data - for sft and preference optimization')
    parser.add_argument('--model-name', type=str, default=None, help="huggingface model-id e.g., meta-llama/Llama-2-7b-hf")

    ## Specific to preference optimization
    parser.add_argument('--beta', default=0.1, type=float)  
    parser.add_argument('--ref-model-path', default=None)

    ## Specific to PEFT
    parser.add_argument('--peft-config-r', default=16, type=int)
    parser.add_argument('--peft-config-lora-alpha', default=48, type=float)
    parser.add_argument('--peft-config-lora-dropout', default=0.05, type=float)
    parser.add_argument('--use-peft', type=bool, default=True)
    parser.add_argument('--max-length', default=128, type=int)

    ## specific to KTO
    parser.add_argument('--desirable-weight', default=1.0, type=float)
    parser.add_argument('--undesirable-weight', default=1.0, type=float)

    ## Specific to FIPO
    parser.add_argument('--lambda-value', default=0.3, type=float)
    parser.add_argument('--weighting-scheme', default='frequency', type=str, help='frequency or uniform')


    ## Specific to PPO
    parser.add_argument('--reward-model-path')

    ## Training arguments
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
    parser.add_argument('--output-dir', default='models', type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    args.train_using = args.train_using.lower()
    
    ref_model_path = args.ref_model_path 
    model_name = args.model_name
    if model_name is not None:
        is_encoder_decoder = 't5' in model_name.lower()
    elif ref_model_path is not None:
        is_encoder_decoder = 't5' in ref_model_path.lower()

    if ref_model_path is None:
        assert model_name is not None, "--model-name is required if ref-model-path is not provided - model-name is the huggingface model-id to use"
        args.output_dir = f'{args.output_dir}/{args.train_using}_{model_name.split("/")[-1]}'
        if is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

        if args.use_peft:
            task_type = TaskType.CAUSAL_LM if not is_encoder_decoder else TaskType.SEQ_2_SEQ_LM
            peft_config = LoraConfig(
                task_type=task_type, 
                inference_mode=False, 
                r=args.peft_config_r, 
                lora_alpha=args.peft_config_lora_alpha, 
                lora_dropout=args.peft_config_lora_dropout
                )
            model = get_peft_model(model, peft_config=peft_config)
            model.print_trainable_parameters()

            if args.train_using == 'fipo':
                model.classification_head = nn.Linear(model.config.hidden_size, len(CLASSES), device=model.device)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token=tokenizer.unk_token
    else:
        model_name = ref_model_path.split('sft_')[-1].strip()
        args.output_dir = f'{args.output_dir}/{args.train_using}_{model_name}'
        if args.train_using != 'ppo':
            if args.use_peft:
                if is_encoder_decoder:
                    model = AutoPeftModelForSeq2SeqLM.from_pretrained(ref_model_path, is_trainable=True, device_map='auto')
                    ref_model = AutoPeftModelForSeq2SeqLM.from_pretrained(ref_model_path, is_trainable=False, device_map='auto')
                else:
                    model = AutoPeftModelForCausalLM.from_pretrained(ref_model_path, is_trainable=True, device_map='auto')
                    ref_model = AutoPeftModelForCausalLM.from_pretrained(ref_model_path, is_trainable=False, device_map='auto')
            else:
                if is_encoder_decoder:
                    model = AutoModelForSeq2SeqLM.from_pretrained(ref_model_path, device_map='auto')
                    ref_model = AutoModelForSeq2SeqLM.from_pretrained(ref_model_path, device_map='auto')
                else:
                    model = AutoModelForCausalLM.from_pretrained(ref_model_path, device_map='auto')
                    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, device_map='auto')
        else:
            if is_encoder_decoder:
                model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(ref_model_path).to('cuda:0')
            else:
                model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_model_path).to('cuda:0')

        tokenizer = AutoTokenizer.from_pretrained(ref_model_path)

    model.enable_input_require_grads()
    if args.use_peft: 
        model.print_trainable_parameters()    

    training_arguments = {
            'output_dir':args.output_dir,               
            'overwrite_output_dir':False,                  
            'num_train_epochs':args.n_epochs,                   
            'per_device_train_batch_size':args.batch_size,         
            'learning_rate':args.learning_rate,                      
            'warmup_steps':args.warmup_steps,                           
            'weight_decay':args.weight_decay,                         
            'adam_epsilon':args.adam_epsilon,                         
            'save_steps':args.save_steps,            
            'eval_steps':4000,          
            'eval_strategy':'steps', 
            'logging_steps':10,                      
            'save_total_limit':1,                         
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
        }

    preference_optimization_args = {
            'beta': args.beta,
            'max_prompt_length':args.max_length,
            'max_length':args.max_length,
            'is_encoder_decoder':is_encoder_decoder,
            'generate_during_eval':True,
    }

    ########################### SFT ###########################
    if args.train_using == 'sft':
        def formatting_prompts_func(example):
            data = []
            for i in range(len(example['prompt'])):
                prompt = example['prompt'][i]
                completion = example['argument'][i]
                text = f"<s> [INST] ### Prompt: {prompt} [/INST] \n### Argument: {completion} </s>"
                data.append(text)      
            return data

        train_data = load_dataset('json', data_files=args.train_data, split='train')
        config = SFTConfig(**training_arguments)
        trainer = SFTTrainer(
            model=model, 
            tokenizer=tokenizer,
            args=config, 
            train_dataset=train_data, 
            formatting_func=formatting_prompts_func
        )
        trainer.train()
        trainer.save_model(args.output_dir)
        return
    
    def map_preference_data(example):
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
        

    if args.train_using == 'ppo':
        def map_ppo_data(sample): 
            sample['query'] = '<s> [INST] ### Prompt: ' + sample['prompt'] + f" [/INST]\n### Argument:"
            return sample
        
        train_data = load_dataset('json', data_files=args.train_data, split='train')
        train_data = train_data.map(map_ppo_data)
        
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        config = PPOConfig(
            model_name=model_name,
            reward_model=args.reward_model_path,
            **training_arguments)
        
        model = AutoModelForCausalLMWithValueHead.from_pretrained(args.ref_model_path).to('cuda:0')
        tokenizer = AutoTokenizer.from_pretrained(args.ref_model_path, padding_side='left')

        reward_model_path = args.reward_model_path
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path).to('cuda:1')
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)

        trainer = PPOTrainer(
            model=model, 
            tokenizer=tokenizer,
            config=config, 
            dataset=train_data, 
        )

        for _ in tqdm(range(trainer.config.ppo_epochs)):
            for _, batch in tqdm(enumerate(trainer.dataloader)):
                queries = batch['query']
                
                tokenized = tokenizer(queries, padding='max_length', max_length=128, truncation=True)

                input_ids = [torch.tensor(ids).to('cuda:0') for ids in tokenized['input_ids']]
                responses = [trainer.generate(ids, return_prompt=False, do_sample=True, max_new_tokens=30, no_repeat_ngram_size=2, pad_token_id=tokenizer.pad_token_id)[0].to('cuda:0') for ids in input_ids]
                batch['response'] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in responses]
            
                tokens = reward_tokenizer(batch['response'], padding=True, truncation=True)
                tokens = {k: torch.tensor(v).to('cuda:1') for k, v in tokens.items()}
                rewards = reward_model(**tokens)
                rewards = rewards.logits[:, 0]
                rewards = [torch.tensor(r).to('cuda:0') for r in rewards]
                
                stats = trainer.step(input_ids, responses, rewards)
                trainer.log_stats(stats, batch, rewards)
                        
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        return


    train_data = load_dataset('json', data_files=args.train_data, split='train') 
    train_data = train_data.map(lambda x: map_preference_data(x))

    if args.train_using == 'dpo':
        print("TRAINING DPO")
        config = DPOConfig(
            **preference_optimization_args,
            **training_arguments)
        trainer = DPOTrainer(
            model=model, 
            ref_model=ref_model,
            tokenizer=tokenizer,
            args=config, 
            train_dataset=train_data, 
        )

    elif args.train_using == 'cpo':
        config = CPOConfig(
            **preference_optimization_args,
            **training_arguments)
        trainer = CPOTrainer(
            model=model, 
            tokenizer=tokenizer,
            args=config, 
            train_dataset=train_data, 
        )

    elif args.train_using == 'kto':
        from datasets import Dataset
        import pandas as pd
        def preference_data_for_kto():
            train_data = load_dataset('json', data_files=args.train_data, split='train') 
            converted_data = []
            seen_completions = []
            for entry in train_data:
                prompt = entry["prompt"]
                chosen_completion = entry["chosen"]
                rejected_completion = entry["rejected"]
                
                chosen_entry = {"prompt": prompt, "completion": chosen_completion, "label": True}
                rejected_entry = {"prompt": prompt, "completion": rejected_completion, "label": False}
                
                if chosen_completion not in seen_completions:
                    converted_data.append(chosen_entry)
                    seen_completions.append(chosen_completion)

                converted_data.append(rejected_entry)
            return Dataset.from_dict(pd.DataFrame(converted_data))
        
        config = KTOConfig(
            **preference_optimization_args,
            desirable_weight=args.desirable_weight,
            undesirable_weight=args.undesirable_weight,
            **training_arguments)
        trainer = KTOTrainer(
            model=model, 
            ref_model=ref_model,
            tokenizer=tokenizer,
            args=config, 
            train_dataset=preference_data_for_kto(), 
        )
    elif args.train_using == 'fipo':
        from collections import Counter
        assert args.weighting_scheme in ['frequency', 'uniform'], "weighting scheme must be either 'frequency' or 'uniform'"
        if args.weighting_scheme == 'frequency':
            fallacy_frequencies = {k: round(v/len(train_data), 3) for k, v in sorted(Counter(train_data['fallacy_type']).items())}
            class_weights = [min(fallacy_frequencies.values())] + list(fallacy_frequencies.values())
        else:
            class_weights = [1/(len(CLASSES)+1)] * (len(CLASSES) + 1)

        config = FIPOConfig(
            **preference_optimization_args,
            **training_arguments)
        trainer = FIPOTrainer(
            model=model, 
            tokenizer=tokenizer,
            args=config, 
            train_dataset=train_data, 
            custom_eval_steps=500,
            clf_loss_class_weights=torch.tensor(class_weights, device=model.device),
            lambda_=args.lambda_value
        )
    
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"FINISHED RUNNING {args.train_using.upper()}, SAVED at", args.output_dir)

if __name__ == '__main__':
    main()