import sys
import os
sys.path.append(os.path.abspath('src/')) 
from utils import save_to
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import transformers
import argparse
import warnings
import torch
warnings.filterwarnings("ignore")

GENERATION_KWARGS = {'max_new_tokens': 50, 'no_repeat_ngram_size': 2, 'do_sample': True, 'top_p': 0.75, 'top_k':10}

data = pd.read_json('/Users/lucamouchel/Downloads/logicalfallacy/Logical-Fallacies/src/baselines/test_data.json')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True, help='llama, mistral, etc...')
    args = parser.parse_args()
    model_id = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    
    generated = []
        
    def generate(prompt: str, model, tokenizer, **generate_kwargs):
        tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=450, truncation=True).to(model.device)
        with torch.no_grad():
            output = model.generate(**tokenized_prompt,
                                    **generate_kwargs,
                                    pad_token_id=tokenizer.eos_token_id)
        output_decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        return output_decoded
    
    
    for _, entry in tqdm(data.iterrows()):
        topic = entry.topic
        stance = 'SUPPORTING' if entry.label == 1 else 'COUNTER'
        prompt = f"""<s> [INST] ### Prompt:  Generate a {stance} argument for the topic: {topic} Consider logical fallacies when generating the argument. Logical fallacies are reasoning errors that make argument seem valid but are in fact logically flawed. Some examples include: 
    - Forcing people to vote is wrong because it's not right to make them do something they shouldn't be forced to do. (Circular Reasoning)
    - I know four poor families. They are lazy drug addicts. Therefore, all poor people are lazy drug addicts. (Faulty Generalization)
    [/INST]\n Now write the argument for the topic: {topic}\n### Argument:
        
        """
        ys = generate(prompt, model, tokenizer, **GENERATION_KWARGS)
        ys = list(map(lambda y : y.split('### Argument:')[-1].strip(), ys))
        y = ys[0] 
        generated.append(y)
        
    save_to(generated, name=f'{model_id}_args.json', output_dir=f'src/baselines/')
  
if __name__ == "__main__":
    main()