from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
import torch
text= "Cannabis should be legal because if you get high you will die."

name = "models/fallacy_clf/howey_electra-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name)

def tokenize(text, tokenizer):
    return tokenizer(text, return_tensors="pt")

inputs = tokenize(text, tokenizer)
with torch.no_grad():
    outputs = model(**inputs)
    print(outputs)