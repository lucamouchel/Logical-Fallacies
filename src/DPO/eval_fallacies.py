import torch

torch.backends.cuda.matmul.allow_tf32 = True
from transformers import T5ForConditionalGeneration
import transformers
from utils import  remove_incomplete_last_sentence, sanitize
import pandas as pd
from tqdm import tqdm
from typing import Optional, Set
from utils import *
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

    
def generate(prompt: str, model, tokenizer: Optional[transformers.PreTrainedTokenizer]):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    tokenized_prompt = tokenizer(prompt, return_tensors='pt', max_length=80, truncation=True).to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids=tokenized_prompt.input_ids, 
                                attention_mask=tokenized_prompt.attention_mask,
                                max_new_tokens=30,
                                #pad_token_id=tokenizer.eos_token_id,
                                #top_p=0.5, 
                                #temperature=1,
                                no_repeat_ngram_size=2,
                                do_sample=False)

        output_decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        return output_decoded

def fallacy_proba(y, clf, clf_tokenizer):
        input_ids = clf_tokenizer.encode(y, add_special_tokens=True, return_tensors='pt')
        input_ids = input_ids.to('cuda')
        with torch.no_grad():
            out = clf(input_ids=input_ids)
            
        logits = out.logits
        fallacy_logit = logits[0][1]        
        fallacy_proba = torch.sigmoid(fallacy_logit).item()
        return fallacy_proba
    
def evaluate_over_dataset(args):
    
    paths = []
    if args.sft_path:
        paths.append(args.sft_path)
    if args.dpo_path:
        paths.append(args.dpo_path)
        
    clf = AutoModelForSequenceClassification.from_pretrained('models/howey_electra-base-mnli', num_labels=2)
    clf.to('cuda')
    clf_tokenizer = AutoTokenizer.from_pretrained('howey/electra-base-mnli')

    for j, path in enumerate(paths):
        print("============================================")
        print(path, '\n')
        ref_model = transformers.AutoModelForCausalLM.from_pretrained(path, device_map='auto')
        #ref_model.to('cuda')
        ref_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(path)
        
        dataset = pd.read_json('data/argumentation/test_cckg.json')
        generated_external = []
        fallacy_probs = []
        for i, entry in tqdm(dataset[:10].iterrows()):
            topic = entry.topic
            stance = entry.label
            prompt = f"<s> [INST] Generate a {'SUPPORTING' if stance==1 else 'COUNTER'} argument for the topic: {topic} [/INST]\n### Argument: "
            generated_ext = generate(prompt, model=ref_model, tokenizer=tokenizer)
            #print('\n')
            if prompt in generated_ext:
                generated_ext = generated_ext[len(prompt) + 1:] #generated_ext.replace(prompt, '')
                
            generated_ext = sanitize(remove_incomplete_last_sentence(generated_ext))
            
            #fallacy_prob = fallacy_proba(generated_ext, clf=clf, clf_tokenizer=clf_tokenizer)
            #print(prompt)
            #print('\n')
            #fallacy_probs.append(fallacy_prob)
            print(generated_ext)

            
            generated_external.append(generated_ext)
 
        
def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--sft-path', default=None)
    parser.add_argument('--dpo-path', default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    evaluate_over_dataset(args)
if __name__ == '__main__':
    main()
