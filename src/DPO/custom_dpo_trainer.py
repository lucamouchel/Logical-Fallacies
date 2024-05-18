from trl import DPOTrainer
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch 
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
import torch.optim as optim
# Inside CustomDPO class __init__ method
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


class CustomDPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super(CustomDPO, self).__init__(*args, **kwargs)
        self.fallacy_clf = AutoModelForSequenceClassification.from_pretrained('models/fallacy_clf/howey_electra-large-mnli', num_labels=2).to('cuda')
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.fallacy_clf.parameters()),
            lr=self.args.learning_rate
        )
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
            fallacy_clf=self.fallacy_clf,
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
            chosen_logits = fallacy_clf(**chosen_inputs).logits[:,1]
            rejected_logits = fallacy_clf(**rejected_inputs).logits[:,1]


            classifier_chosen_scores = F.logsigmoid(chosen_logits).squeeze(-1)  # assuming logits are (batch_size, 1)
            classifier_rejected_scores = F.logsigmoid(rejected_logits).squeeze(-1)  # assuming logits are (batch_size, 1)

            
            ## we want to reduce loss if the chosen score is higher than the rejected score
            ## so we do 1-classifier_chosen_scores + classifier_rejected_scores. 
            # We want the loss to increase if the classifier is confident that the chosen score is a fallacy
            fallacy_penalty = (1 + classifier_chosen_scores) - classifier_rejected_scores
            losses = losses + 0.6*fallacy_penalty
        else:
            classifier_chosen_scores = 1
            classifier_rejected_scores = 1

        ## if the model is confident that the chosen score is a fallacy -- e.g., sigmoid > 0.5, then the reward should be small for the chosen sample - hence 1-sigmoid
        chosen_rewards = (
            (self.beta * (2*classifier_chosen_scores)) ## multiply the chosen rewards by sigmoid of the chosen logits after feeding them through the classifier. This way, we can give more weight to the chosen rewards that the classifier is more confident about.
            * (policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device))
            .detach()
        ) 

        ## if the model is confident that the chosen score is a fallacy -- e.g., sigmoid > 0.5, then the reward should be small for the rejected sample - hence 1-sigmoid
        rejected_rewards = (
            (self.beta * 2* (1-classifier_rejected_scores)) ### multiply the rejected rewards by 1 - sigmoid of the rejected logits after feeding them through the classifier. This way, we can give more weight to the rejected rewards that the classifier is confident about.
            * (policy_rejected_logps.to(self.accelerator.device) - reference_rejected_logps.to(self.accelerator.device))
            .detach()
        )

        return losses, chosen_rewards, rejected_rewards

    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        self.fallacy_clf.train()
        inputs = self._prepare_inputs(inputs)


        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean() 
        
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.optimizer.step()

        return loss.detach() / self.args.gradient_accumulation_steps


