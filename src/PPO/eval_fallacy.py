from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
import torch
import pandas as pd


name = "models/fallacy_clf/howey_electra-large-mnli"
clf_tokenizer = AutoTokenizer.from_pretrained(name)
clf = AutoModelForSequenceClassification.from_pretrained(name)
device='cuda'
ref_model = AutoModelForCausalLM.from_pretrained("models/arguments/sft_Llama-2-7b-hf_trl_2024-03-12_09:54:19.188918", device_map='auto')
model = AutoModelForCausalLM.from_pretrained("models/arguments/ppo__2024-04-01_11:17:28.735799", device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("models/arguments/ppo__2024-04-01_11:17:28.735799")

test_data = pd.read_json('data/argumentation/test_cckg.json')

for i, entry in test_data.iterrows():
    topic = entry.topic
    stance = 'SUPPORTING' if entry.label == 1 else 'COUNTER'
    prompt = f"<s> [INST] ###Prompt:  Generate a {stance} argument for the topic: {topic} [/INST]\n### Argument: "
    
    tokenized = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        
        ref_model_output = ref_model.generate(**tokenized, max_length=128, do_sample=True, no_repeat_ngram_size=2)
        model_output = model.generate(**tokenized, max_length=128, do_sample=True, no_repeat_ngram_size=2)
        print("GENERATION DONE")
        model_decoded = tokenizer.decode(model_output[0], skip_special_tokens=True)
        ref_model_decoded = tokenizer.decode(ref_model_output[0], skip_special_tokens=True)
        
        
        ref_argument = ref_model_decoded.split('### Argument: ')[-1]
        model_argument = model_decoded.split('### Argument: ')[-1]
        clf_input_ref = clf_tokenizer(ref_argument, return_tensors="pt")
        clf_input = clf_tokenizer(model_argument, return_tensors="pt")

        
        ref_output = clf(**clf_input_ref)
        output = clf(**clf_input)
        
        ref_rewards = ref_output.logits[:, 0]
        model_rewards = output.logits[:, 0]
        
    print("REF MODEL")
    print(ref_model_decoded)
    print(ref_rewards)

    print()
    print("MODEL")
    print(model_decoded)
    print(model_rewards)   
    print("\n---------\n") 
    
    
