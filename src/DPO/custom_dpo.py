from trl import DPOTrainer
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch 
from transformers import BertModel, BertTokenizer
import torch.nn as nn
from contextlib import contextmanager, nullcontext

from transformers import PreTrainedModel

class FallacyClassifier(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout):
        super(FallacyClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )  
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Use the pooled output for classification
        return self.classifier(pooled_output)

    
hidden_dim = 256
output_dim = 13
dropout = 0.5

criterion = nn.CrossEntropyLoss()
classifier = FallacyClassifier(hidden_dim, output_dim, dropout).to('cuda')

class CustomDPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super(CustomDPO, self).__init__(*args, **kwargs)
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
     
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext


        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        input_ids = inputs['rejected_input_ids'].to(self.args.device)
        attention_mask = inputs['rejected_attention_mask'].to(self.args.device)
        rejected_labels = inputs['rejected_labels'].to(self.args.device)

        outs = classifier(input_ids, attention_mask)
        clf_loss = criterion(outs, rejected_labels)
        clf_loss.backward()


        lambda_ = 0.2
        print(loss) 
        loss = loss.to(self.args.device) + (lambda_ * clf_loss).to(self.args.device)
        print(loss)
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss