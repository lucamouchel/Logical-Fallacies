from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration, RagTokenizer, pipeline
import torch
from datasets import load_dataset
import json 
# only import a couple of samples
print("LOADING")
from tqdm import tqdm
rag_dataset = load_dataset("wiki_dpr", 'psgs_w100.multiset.compressed', split='train', cache_dir='cache', trust_remote_code=True)
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-nq", indexed_dataset=rag_dataset) 
rag_model = RagSequenceForGeneration.from_pretrained('facebook/rag-token-nq', indexed_dataset=rag_dataset) 
rag_tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
summarizer = pipeline('summarization', max_length = 80)

def get_summary(prompt):
    inputs = rag_tokenizer(prompt, max_length=80, padding=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    question_hidden_states = rag_model.question_encoder(input_ids)[0] 
    docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors='pt')
    doc_scores = torch.bmm(
            question_hidden_states.unsqueeze(1),
            docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
    ).squeeze(1)

    doc_ids = docs_dict["doc_ids"]
    summary = summarizer(rag_dataset[doc_ids[0]]['text'], max_length=80)
    summaries = [x['summary_text'] for x in summary]
    return summaries[0]

with open('data/sft/arguments/train.json', 'r') as f:
    data = json.load(f) 

new_data = []
for item in tqdm(data, total=len(data)):
    try:
        prompt = item['prompt']
        topic = prompt.split('topic: ')[-1].strip()
        argument = item['argument']
        context = get_summary(topic).strip()
        sample = {'prompt': prompt, 'context': context, 'argument': argument}
        new_data.append(sample)
    except:        
        with open('data/sft_rag/arguments/train.json', 'w') as f:
            json.dump(new_data, f, indent=4)
  
with open('data/sft_rag/arguments/train.json', 'w') as f:
    json.dump(new_data, f, indent=4)
