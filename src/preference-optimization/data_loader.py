from datasets import Dataset, load_dataset
from collections import Counter
from typing import List
import pandas as pd 
import argparse 
import torch


def sft_formatting_function(data) -> List[str]:
    res = []
    for i in range(len(data['prompt'])):
        prompt = data['prompt'][i]
        completion = data['argument'][i]
        text = f"<s> [INST] ### Prompt: {prompt} [/INST] \n### Argument: {completion} </s>"
        res.append(text)      
    return res

def sample_classes(dataset: Dataset, class_column: str='fallacy_type', num_samples: int=200) -> Dataset:
    import random
    random.seed(42)
    sampled_indices = []
    class_counts = dataset[class_column]
    unique_classes = set(class_counts)
    for cls in unique_classes:
        indices = [i for i, x in enumerate(class_counts) if x == cls]
        sampled_indices.extend(random.sample(indices, min(len(indices), num_samples), ))
    sampled_indices.sort()
    return dataset.select(sampled_indices)
        
def map_preference_data_kto(data) -> Dataset:
    converted_data = []
    seen_completions = []
    for entry in data:
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


def map_preference_data(data) -> dict:
    prompt = data['prompt']
    if 'supporting argument' in prompt:
        prompt = prompt.replace('supporting argument', 'SUPPORTING argument')
    elif 'counter argument' in prompt:
        prompt = prompt.replace('counter argument', 'COUNTER argument')

    return {
            'prompt': '<s> [INST] ### Prompt: ' + prompt + f" [/INST]\n### Argument: " ,
            'chosen': data['chosen'] + ' </s>',
            'rejected': data['rejected'] +  " </s>",
            'fallacy_type': data['fallacy_type']
        }

class DataLoader:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.data = load_dataset('json', data_files=args.train_data, split='train')
        self.config_kwargs = {}
        self.trainer_kwargs = {}
    def load_data(self, train_using: str) -> None:
        if self.args.uniform_data:
            self.data = sample_classes(self.data, num_samples=200)
        
        if train_using == 'sft':
            self.trainer_kwargs['formatting_func'] = sft_formatting_function
        elif train_using in ['cpo', 'dpo']:
            self.data = self.data.map(map_preference_data)
        elif train_using == 'kto':
            self.data = map_preference_data_kto(self.data)
            self.config_kwargs['desirable_weight'] = self.args.desirable_weight
            self.config_kwargs['undesirable_weight'] = self.args.undesirable_weight
        elif train_using == 'fipo':
            assert self.args.weighting_scheme in ['frequency', 'uniform'], "weighting scheme must be either 'frequency' or 'uniform'"
            self.data = self.data.map(map_preference_data)
            if self.args.weighting_scheme == 'frequency':
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                fallacy_frequencies = {k: round(v/self.__len__(), 3) for k, v in sorted(Counter(self.__getitem__('fallacy_type')).items())}
                class_weights = torch.tensor([min(fallacy_frequencies.values())] + list(fallacy_frequencies.values()), device=device)
            else:
                class_weights = None
            self.trainer_kwargs['custom_eval_steps'] = 500
            self.trainer_kwargs['clf_loss_class_weights'] = class_weights
            self.trainer_kwargs['lambda_'] = self.args.lambda_value
        else: 
            raise ValueError(f"train_using must be one of ['sft', 'dpo', 'cpo', 'kto', 'fipo', 'ppo'] - got {train_using}")
            

    def __getdata__(self) -> Dataset:
        return self.data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> dict:
        return self.data[idx]
 
    def __configkwargs__(self) -> dict:
        return self.config_kwargs
    
    def __trainerkwargs__(self) -> dict:
        return self.trainer_kwargs