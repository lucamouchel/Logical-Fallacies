import pandas as pd 
import torch
import transformers 
import json
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
clf = transformers.AutoModelForSequenceClassification.from_pretrained('models/fallacy_clf/howey_electra-large-mnli', num_labels=2)
tokenizer = transformers.AutoTokenizer.from_pretrained('models/fallacy_clf/howey_electra-large-mnli')
clf.eval()
    
def fallacy_proba(y):
    input_ids = tokenizer.encode(y, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        out = clf(input_ids=input_ids)
        
    logits = out.logits
    fallacy_logit = logits[0][1]        
    fallacy_proba = torch.sigmoid(fallacy_logit).item()
    return fallacy_proba


with open('results/llama/arguments/dpo/f-rate.json', 'r') as f:
    data = json.load(f)


probas_pred = []
gpt_pred = []
for i, entry in tqdm(enumerate(data), total=len(data)):
    argument = entry['argument']
    fallacy_type = entry['fallacy_type']

    prob = fallacy_proba(argument)

    if fallacy_type == "None":
        gpt_pred.append(0)
        pred=0
    else: 
        gpt_pred.append(1)
        pred = 1

    if prob >= 0.7:
        probas_pred.append(1)
        clf_pred = 1
    else:
        probas_pred.append(0)
        clf_pred = 0

    # if clf_pred != pred:
    #     print(prob)
    #     print(probas_pred[-1])
    #     print(argument)
    #     print("***********")

print(cohen_kappa_score(probas_pred, gpt_pred))