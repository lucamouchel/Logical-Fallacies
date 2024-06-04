from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch

data = load_dataset('json', data_files='data/dpo/arguments/train.json', split='train') 

dataset = []
chosen_samples = set()
for entry in data:
    instruction = entry['prompt']
    chosen = entry['chosen']
    rejected = entry['rejected']
    if chosen not in chosen_samples:
        chosen_samples.add(chosen)
        dataset.append({'text': instruction + ' ' + chosen, 'label': 0})
    
    dataset.append({'text': instruction + ' ' + rejected, 'label': entry['fallacy_type']})

print(len(dataset))

# Initialize the tokenizer and model
model_name = 'howey/electra-large-mnli'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=14, ignore_mismatched_sizes=True).to('cuda')  # 13 fallacies + 1 valid

class FallacyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']

        encoded = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'label': label
        }
    
# Load your data here
# data = [{'prompt': ..., 'chosen': ..., 'rejected': ..., 'fallacy_type': ...}, ...]



dataset = FallacyDataset(dataset, tokenizer)

train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32)

model.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
device='cuda'

def train(model, train_loader, val_loader, criterion, optimizer, epochs=3):
    best_val_loss = 1000
    for epoch in tqdm(range(epochs)):
        model.train()
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Train on chosen samples
            outs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outs.loss
           

            if i % 10 == 0:
                print("LOSS", loss.item())

            loss.backward()
            optimizer.step()
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                text = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                outs = model(input_ids=text, attention_mask=batch['attention_mask'].to(device), labels=labels)
                val_loss=outs.loss
                total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss/len(val_loader)
            print(f'Val Loss: {total_val_loss/len(val_loader)}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model
            model.save_pretrained('models/fallacy_classifierX')
            tokenizer.save_pretrained('models/fallacy_classifierX')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 3:
                print("Early stopping triggered")
                break
                
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

train(model, train_loader, val_loader, criterion, optimizer)

# Save the model
model.save_pretrained('models/fallacy_classifierX')
tokenizer.save_pretrained('models/fallacy_classifierX')

def predict_fallacy(prompt, argument, model, tokenizer):
    # Combine the prompt and argument
    input_text = prompt + " " + argument
    
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Get the logits from the classifier
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    
    # Predict the fallacy type
    predicted_label = torch.argmax(logits, dim=-1).item()
    
    # Return the predicted fallacy type
    return predicted_label