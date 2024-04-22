from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification


fallacy_classifier = transformers.AutoModelForSequenceClassification.from_pretrained(arguments.clf_path, num_labels=2)
clf_tokenizer = transformers.AutoTokenizer.from_pretrained(arguments.clf_path)

def fallacy_proba(y, return_extra):
        input_ids = clf_tokenizer.encode(y, add_special_tokens=True, return_tensors='pt')
        with torch.no_grad():
            out = clf(input_ids=input_ids)
            
        logits = out.logits[0]
        fallacy_logit = logits[1]        
        non_fallacy_logits = logits[0]
        fallacy_proba = torch.sigmoid(fallacy_logit).item()
        non_fallacy_proba = torch.sigmoid(non_fallacy_logits).item()
        if return_extra:
            return non_fallacy_proba, fallacy_proba
        return fallacy_proba

with open('results/llama/arguments/sft/f-rate.json', 'r') as f:
    preds = json.load(f)

for i, sample in tqdm(enumerate(preds), total=len(test)):
        fallacy_pred = sample['fallacy_type']
        argument = sample['argument']
    
        non_fallacy, fallacy = fallacy_proba(argument, return_extra=True)
        if fallacy > 0.5:
            print(fallacy)
            print(non_fallacy)
            print(fallacy_pred)
            sample = {'prompt': prompt, 'chosen': chosen, 'rejected': y}
            new_dataset.append(sample)
            print(sample)
        total_probas.append(proba)

