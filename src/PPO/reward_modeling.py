from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from trl import RewardConfig, RewardTrainer
import argparse
from datetime import datetime
from peft import LoraConfig, TaskType
import wandb
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/dpo/')
    parser.add_argument('--reward-model-name', required=True)
    parser.add_argument('--n-epochs', default=10, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=2, type=int)
    parser.add_argument('--learning-rate', default=2e-4, type=float)
    parser.add_argument('--warmup-steps', default=100, type=int)
    parser.add_argument('--weight-decay', default=0.05, type=float)
    parser.add_argument('--adam-epsilon', default=1e-8, type=float)
    parser.add_argument('--save-steps', default=500, type=int)
    parser.add_argument('--logging-steps', default=50, type=int)
    parser.add_argument('--output-dir', default='models')
    parser.add_argument('--max-length', default=512, type=int)
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
    
def tokenize(sample, tokenizer):
    prompt = '<s> [INST] ### Prompt: ' + sample['prompt'] + '[/INST]\n### Argument: '
    chosen = prompt + sample['chosen'] + ' </s>'
    rejected = prompt + sample['rejected'] + ' </s>'

    chosen_tokenized = tokenizer(chosen)
    reject_tokenized = tokenizer(rejected)
    return {
            'input_ids_chosen': chosen_tokenized.input_ids,
            'attention_mask_chosen': chosen_tokenized.attention_mask,
            'input_ids_rejected': reject_tokenized.input_ids,
            'attention_mask_rejected':  reject_tokenized.attention_mask
            }

def main():
    args = parse_args()
    train_data = load_dataset('json', data_files='data/dpo/arguments/train.json', split='train')

    ####################################################################
    ######################## REWARD MODELING ###########################

    model_name = args.reward_model_name
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05)
    training_args = get_training_args(args)
    reward_config = RewardConfig(**training_args.to_dict(), max_length=args.max_length)
    train_data = train_data.map(lambda sample: tokenize(sample, tokenizer))
    
    reward_trainer = RewardTrainer(
                model=model,
                tokenizer=tokenizer,
                args=reward_config,
                train_dataset=train_data,
            )

    reward_trainer.train()
    if '/' in model_name:
        output_directory =f'models/arguments/reward_model_{model_name.split("/")[-1]}_{datetime.now()}'
    else:
        output_directory =f'models/arguments/reward_model_{model_name}_{datetime.now()}'

    args.output_dir = output_directory.replace(' ', '_')
    reward_trainer.save_model(args.output_dir)

    ####################################################################
    ####################################################################
    ####################################################################

if __name__ == '__main__':
    main()