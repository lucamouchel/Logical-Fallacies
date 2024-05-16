from trl import DPOTrainer
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch 
from transformers import BertModel, BertTokenizer
import torch.nn as nn
from contextlib import contextmanager, nullcontext
import torch.nn.functional as F
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
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

    
hidden_dim = 256
output_dim = 1
dropout = 0.5

criterion = nn.CrossEntropyLoss()
classifier = FallacyClassifier(hidden_dim, output_dim, dropout).to('cuda')

class CustomDPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super(CustomDPO, self).__init__(*args, **kwargs)
        self.fallacy_clf = classifier

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            fallacy_clf=classifier,
            chosen_inputs={'input_ids': batch['chosen_input_ids'], 'attention_mask': batch['chosen_attention_mask']},
            rejected_inputs={'input_ids': batch['rejected_input_ids'], 'attention_mask': batch['rejected_attention_mask']}
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean(), metrics

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        fallacy_clf: nn.Module = None,
        chosen_inputs=None,
        rejected_inputs=None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )
        
        ####### MODIFS HERE
        if fallacy_clf:
            chosen_logits = fallacy_clf(**chosen_inputs)
            rejected_logits = fallacy_clf(**rejected_inputs)
            criterion = nn.BCEWithLogitsLoss()
            binary_labels = torch.cat([torch.ones_like(chosen_logits), torch.zeros_like(rejected_logits)], dim=0)
            chosen_logits_flat = chosen_logits.view(-1)
            
            rejected_logits_flat = rejected_logits.view(-1)
            binary_labels_flat = binary_labels.view(-1)

            # Compute binary classification loss
            binary_classification_loss = criterion(
                torch.cat([chosen_logits_flat, rejected_logits_flat], dim=0), binary_labels_flat
            )

            losses = losses.mean()
            losses = losses + binary_classification_loss ## add losses together

            classifier_chosen_scores = torch.sigmoid(chosen_logits).detach().squeeze(-1) 
            classifier_rejected_scores = torch.sigmoid(rejected_logits).detach().squeeze(-1) 

        chosen_rewards = (
            (self.beta * classifier_chosen_scores) ## multiply the chosen rewards by sigmoid of the chosen logits after feeding them through the classifier. This way, we can give more weight to the chosen rewards that the classifier is more confident about.
            * (policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device))
            .detach()
        ) 
        rejected_rewards = (
            (self.beta * (1-classifier_rejected_scores)) ### multiply the rejected rewards by 1 - sigmoid of the rejected logits after feeding them through the classifier. This way, we can give more weight to the rejected rewards that the classifier is confident about.
            * (policy_rejected_logps.to(self.accelerator.device) - reference_rejected_logps.to(self.accelerator.device)).detach()
        )

        print(losses)
        print(chosen_rewards)
        print(rejected_rewards)

        return losses, chosen_rewards, rejected_rewards
