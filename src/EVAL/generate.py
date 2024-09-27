import sys
import os
sys.path.append(os.path.abspath('src/')) 
from peft import AutoPeftModelForCausalLM, PeftModel
from utils import save_to
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import transformers
import argparse
import warnings
import torch
warnings.filterwarnings("ignore")

GENERATION_KWARGS = {'max_new_tokens': 50, 'no_repeat_ngram_size': 2, 'do_sample': True, 'top_p': 0.75, 'top_k':15}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--model-name', required=True, help='llama, mistral, etc...')
    parser.add_argument('--type', required=True, help='sft, dpo, cpo, kto,..')
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', device_map='auto')
    #model = PeftModel.from_pretrained(model, args.model_path)
    #model = model.merge_and_unload()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    test_set = pd.read_json('data/argumentation/test_cckg.json')
    
    def generate(prompt: str, model, tokenizer, n=5, **generate_kwargs):
        tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=200, truncation=True).to(model.device)
        with torch.no_grad():
            output = model.generate(**tokenized_prompt,
                                    **generate_kwargs,
                                    pad_token_id=tokenizer.eos_token_id,
                                    num_return_sequences=n)
        output_decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        return output_decoded
    
    arguments = []
    for _, entry in tqdm(test_set[:200].iterrows(), total=len(test_set)):
        topic = entry.topic
        stance = 'SUPPORTING' if entry.label == 1 else 'COUNTER'
        prompt = f"<s> [INST] ### Prompt:  Generate a {stance} argument for the topic: {topic} [/INST]\n### Argument:"
        ys = generate(prompt, model, tokenizer, n=1, **GENERATION_KWARGS)
        ys = list(map(lambda y : y.split('### Argument:')[-1].strip(), ys))
        y = ys[0] 
        arguments.append(y)
        
    save_to(arguments, name=f'{args.type}_args.json', output_dir=f'results/{args.model_name}/')
  
if __name__ == "__main__":
    main()