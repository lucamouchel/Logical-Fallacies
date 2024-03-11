import json

with open('/root/logical-fallacy-LLMs/data/dpo/arguments/test.json', 'r') as f:
    train = json.load(f)
    
print(train.keys())

data = []
for i, prompt in enumerate(train['prompt']):
    d = {'prompt': prompt, 'chosen': train['chosen'][i], 'rejected': train['rejected'][i]}
    data.append(d)
    
with open('/root/logical-fallacy-LLMs/data/dpo/arguments_dpo/test.json', 'w') as f:
    json.dump(data, f, indent=4)