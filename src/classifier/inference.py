import transformers 

# SCRIPT TO RUN INFERENCE ON A SINGLE MODEL FOR MULTICLASS CLASSIFICATION
# This script is used to run inference on a single model for multiclass classification

# Importing necessary libraries
import os
import torch
import numpy as np

CLASSES = {'Not a Fallacy': 0,  'faulty generalization': 1, 'false causality': 2, 'fallacy of relevance': 3, 'fallacy of extension': 4, 'equivocation': 5, 'ad populum': 6, 'appeal to emotion': 7, 'ad hominem': 8, 'circular reasoning': 9, 'fallacy of credibility': 10, 'fallacy of logic': 11, 'false dilemma': 12, 'intentional': 13}
INVERSE_CLASSES = {v: k for k, v in CLASSES.items()}
model = transformers.AutoModelForSequenceClassification.from_pretrained('models/fallacy_classifier')
tokenizer = transformers.AutoTokenizer.from_pretrained('models/fallacy_classifier')

def predict(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.detach().numpy()

def predict_fallacy(prompt, argument):
    input_text = prompt + " " + argument
    probs = predict(input_text)
    predicted_label = np.argmax(probs)
    # for key, value in CLASSES.items():
    #     if value == predicted_label:
    #         predicted_label = key
    return predicted_label

import pandas as pd
from tqdm import tqdm
accuracy= 0
test_data = pd.read_json('data/generated/test_cckg.json')

test_data = pd.read_json('results/llama/arguments/sft/f-rate.json')

for i, entry in tqdm(test_data.iterrows(), total=len(test_data)):
    prompt = "Generate a supporting argument for the topic: "+entry['topic']
    argument = entry['argument']
    fallacy_type=  entry['fallacy_type']
    if fallacy_type == 'None':
        fallacy_type = 'Not a Fallacy'
    predicted_fallacy = predict_fallacy(prompt, argument)
    
    print(fallacy_type, predicted_fallacy)

    if predicted_fallacy == CLASSES[fallacy_type]:
        accuracy += 1
        print(accuracy)
accuracy = accuracy/len(test_data)
print(accuracy)