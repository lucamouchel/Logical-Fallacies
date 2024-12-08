from transformers import AutoTokenizer, PreTrainedModel, TrainingArguments, Trainer
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset, Dataset
from FIPO.FIPOTrainer import FIPOTrainer
from FIPO.FIPOConfig import FIPOConfig
from data_loader import DataLoader
from model import LoadModels
import torch.nn as nn
import pandas as pd
import argparse
import PO_utils 
import pathlib
import torch
import sys 
import ppo
sys.path.append('src/')

class Trainer():
    def __init__(self, 
                 model: PreTrainedModel, 
                 ref_model: Optional[PreTrainedModel], 
                 tokenizer: AutoTokenizer, 
                 args: argparse.Namespace,
                 trainer_cls: Trainer,
                 config_cls: TrainingArguments,
                 train_dataset: Dataset,
                 is_encoder_decoder: bool=False,
                 config_kwargs: Dict = {},
                 training_kwargs: Dict = {}) -> None: 
        
        self.model=model
        self.ref_model=ref_model
        self.tokenizer=tokenizer
        self.args=args
        self.trainer_cls=trainer_cls
        self.config_cls=config_cls
        self.train_dataset=train_dataset
        self.config_kwargs=config_kwargs
        self.training_kwargs=training_kwargs
        
        self.training_arguments = {
            'output_dir':args.output_dir,               
            'overwrite_output_dir': False,                  
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

        self.preference_optimization_args = {
            'beta': args.beta,
            'max_prompt_length':args.max_length,
            'max_length':args.max_length,
            'is_encoder_decoder': is_encoder_decoder,
            'generate_during_eval':True,
        }
                
        self.config, self.trainer = self.init_config_trainer()

    def init_config_trainer(self) -> Tuple[TrainingArguments, Trainer]:
        if self.args.train_using != 'sft':
            config = self.config_cls(**self.training_arguments, **self.preference_optimization_args, **self.config_kwargs)
            trainer = self.trainer_cls(**{'model': self.model, 'ref_model': self.ref_model, 'tokenizer': self.tokenizer, 'args': config, 'train_dataset': self.train_dataset}, 
                                       **self.training_kwargs)
        else:
            config = self.config_cls(**self.training_arguments, 
                                     **self.config_kwargs)
            trainer = self.trainer_cls(**{'model': self.model, 'tokenizer': self.tokenizer, 'args': config, 'train_dataset': self.train_dataset}, 
                                       **self.training_kwargs)
        return config, trainer
        
    def train_(self) -> None:
            self.trainer.train()
            self.trainer.save_model(self.args.output_dir)
        
        
def main() -> None:
    args = PO_utils.parse_args()
    args.train_use = args.train_using.lower()
    
    if args.train_using == 'ppo':
        ppo.train(args)
        return
    
    loaded_models = LoadModels(args=args)
    data_loader = DataLoader(args)
    data_loader.load_data(args.train_using)
    
    config_cls, trainer_cls = PO_utils.get_trainer_and_config_cls(args.train_using)
    TRAINER = Trainer(
        model=loaded_models.model, 
        ref_model=loaded_models.ref_model, 
        tokenizer=loaded_models.tokenizer, 
        args=args, 
        trainer_cls=trainer_cls, 
        config_cls=config_cls, 
        is_encoder_decoder=loaded_models.is_encoder_decoder,
        train_dataset=data_loader.__getdata__(), 
        config_kwargs=data_loader.__configkwargs__(),
        training_kwargs=data_loader.__trainerkwargs__()
    )
    
    TRAINER.train_()
if __name__ == '__main__':
    main()