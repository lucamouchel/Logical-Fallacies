from trl import DPOTrainer
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch 
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
import torch.optim as optim
# Inside CustomDPO class __init__ method
import wandb
CLASSES = {'Not a Fallacy': 0,  'faulty generalization': 1, 'false causality': 2, 'fallacy of relevance': 3, 'fallacy of extension': 4, 'equivocation': 5, 'ad populum': 6, 'appeal to emotion': 7, 'ad hominem': 8, 'circular reasoning': 9, 'fallacy of credibility': 10, 'fallacy of logic': 11, 'false dilemma': 12, 'intentional': 13}
wandb.init(mode="disabled") 
class FallacyClassifier(torch.nn.Module):
    def __init__(self):
        super(FallacyClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(32000, 32000),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(32000, 14)
        )  
        
    def forward(self, x):
        outputs = torch.mean(x, dim=1)
        outputs = self.classifier(outputs)
        return outputs 

fallacy_classifier = FallacyClassifier().to('cuda')

class CustomDPO(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super(CustomDPO, self).__init__(*args, **kwargs)
        self.fallacy_clf = fallacy_classifier
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.fallacy_clf.parameters()),
            lr=self.args.learning_rate
        )
        
    def get_batch_loss_metrics(self, model, batch: Dict[str, Union[List, torch.LongTensor]], train_eval: Literal["train", "eval"] = "train"):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        policy_chosen_logps,policy_rejected_logps, policy_chosen_logits, policy_rejected_logits = self.concatenated_forward(model, batch)

        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(self.model, batch)
                else:
                    reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(self.ref_model, batch)


        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            fallacy_clf=self.fallacy_clf,
            chosen_inputs=policy_chosen_logits,
            rejected_inputs=policy_rejected_logits,
            fallacy_types=batch['fallacy_type']    
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
        fallacy_types=None
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
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )
        
        ####### MODIFS HERE
        if fallacy_clf:
            chosen_clf = fallacy_classifier(chosen_inputs)
            rejected_clf = fallacy_classifier(rejected_inputs)

            rejected_targets = torch.tensor(fallacy_types).view(-1).to('cuda')
            chosen_targets = torch.zeros(chosen_clf.size(0), dtype=torch.long).to('cuda')

            concatenated_outputs = torch.cat([chosen_clf, rejected_clf], dim=0)
            concatenated_targets = torch.cat([chosen_targets, rejected_targets], dim=0)
            #clf_loss = F.cross_entropy(concatenated_outputs, concatenated_targets)
            ## the whole loss makes model crash - cuda out of memory
            clf_loss = F.cross_entropy(rejected_clf, rejected_targets)
            # chosen_loss = F.cross_entropy(chosen_clf, chosen_targets)


            # clf_loss = chosen_loss + rejected_loss
            print(losses)
            losses = losses + 0.1*clf_loss
            print(losses)
            
            
            chosen_confidence = F.softmax(chosen_clf, dim=1)[:, 0]
            rejected_confidence = F.softmax(rejected_clf, dim=1)[range(rejected_clf.size(0)), torch.tensor(fallacy_types).view(-1).to('cuda')]
            ## both confidence scores will increase over time -- we must do 1-classifier_rejected_scores so that the rejected rewards decrease over time
            classifier_chosen_scores = chosen_confidence 
            classifier_rejected_scores = 1-rejected_confidence

        else:
            classifier_chosen_scores = 1
            classifier_rejected_scores = 1

        ## if the model is confident that the chosen score is a fallacy -- e.g., sigmoid > 0.5, then the reward should be small for the chosen sample - hence 1-sigmoid
        chosen_rewards = (
            (self.beta ) ## multiply the chosen rewards by sigmoid of the chosen logits after feeding them through the classifier. This way, we can give more weight to the chosen rewards that the classifier is more confident about.
            * (policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device))
            .detach()
        ) * 4*classifier_chosen_scores
        
       
        ## if the model is confident that the chosen score is a fallacy -- e.g., sigmoid > 0.5, then the reward should be small for the rejected sample - hence 1-sigmoid
        rejected_rewards = (
            (self.beta ### multiply the rejected rewards by 1 - sigmoid of the rejected logits after feeding them through the classifier. This way, we can give more weight to the rejected rewards that the classifier is confident about.
            * (policy_rejected_logps.to(self.accelerator.device) - reference_rejected_logps.to(self.accelerator.device))
            .detach()
       )) * 4*classifier_rejected_scores
        
        print("*"*50)
        return losses, chosen_rewards, rejected_rewards

    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()

        self.optimizer.zero_grad()
    
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean() 
            
        print(loss)

        self.accelerator.backward(loss)

        total_params = 0
        updated_params = 0
        
        params = (fallacy_classifier.named_parameters())
        for name, param in params:
            if param.grad is not None:
                print(f'{name} gradient: {param.grad.sum()}')
            else:
                print(f'{name} has no gradient')
        
        self.optimizer.step()
        # self.fallacy_clf.zero_grad()

        return loss.detach() / self.args.gradient_accumulation_steps


