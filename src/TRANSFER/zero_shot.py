from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import argparse
import pandas as pd 
from tqdm import tqdm
import sys 
sys.path.append('src/')
from EVAL.utils import generate
from DPO.utils import save_to, remove_incomplete_last_sentence
import pathlib

GENERATION_KWARGS = {'max_new_tokens': 40, 'no_repeat_ngram_size': 2, 'do_sample': True}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', help="Hugging face model card name", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    test_set = pd.read_json('data/argumentation/test_cckg.json')
    
    generated = []
    for i, entry in tqdm(test_set.iterrows(), total=len(test_set)):
        topic = entry.topic
        stance = 'SUPPORTING' if entry.label == 1 else 'COUNTER'
        prompt = f"<s> [INST] ### Prompt:  Generate a 15 word {stance} argument for the topic: {topic} [/INST]\n### Argument: "
        
        with torch.no_grad():
            y_model = generate(prompt, model, tokenizer, **GENERATION_KWARGS)
            y_model = remove_incomplete_last_sentence(y_model.split('### Argument:')[-1].strip())
            sample = {'topic': topic, 'stance': stance, 'generated': y_model}
            generated.append(sample)
        
    save_to(generated, name=f'generated.json', output_dir=f'results/{args.model_id.split('/')[-1]}/')

if __name__ == "__main__":
    main()