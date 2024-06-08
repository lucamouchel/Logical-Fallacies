from trl import DPOTrainer
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch 
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoTokenizer
import torch.optim as optim
from utils import remove_incomplete_last_sentence
from trl.models import create_reference_model
# Inside CustomDPO class __init__ method
import wandb
from copy import deepcopy
from  transformers.utils import is_sagemaker_mp_enabled
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward
CLASSES = {'Not a Fallacy': 0,  'faulty generalization': 1, 'false causality': 2, 'fallacy of relevance': 3, 'fallacy of extension': 4, 'equivocation': 5, 'ad populum': 6, 'appeal to emotion': 7, 'ad hominem': 8, 'circular reasoning': 9, 'fallacy of credibility': 10, 'fallacy of logic': 11, 'false dilemma': 12, 'intentional': 13}



#fallacy_classifier = AutoModelForSequenceClassification.from_pretrained('models/fallacy/clf').to('cuda')
#fallacy_tokenizer = AutoTokenizer.from_pretrained('models/fallacy/clf')
class CustomDPO(DPOTrainer):
    def __init__(self, custom_eval_steps, lambda_, *args, **kwargs):
        super(CustomDPO, self).__init__(*args, **kwargs)
        self.ref_model = deepcopy(self.model)
        self.lambda_ = lambda_
        self.ref_model.print_trainable_parameters()
        for _, param in self.ref_model.named_parameters():
            param.requires_grad = False
        self.ref_model = self.ref_model.eval()
       

        #self.fallacy_clf = fallacy_classifier
        #self.fallacy_tokenizer = fallacy_tokenizer
        self.custom_eval_steps = custom_eval_steps
        self.current_train_steps = 0
        self.optimizer2 = optim.Adam(
             list(self.model.classification_head.parameters()),
             lr=5e-4
         )
    
    def print_inference(self, batch):
        if self.current_train_steps % self.custom_eval_steps == 0:
            prompt_input_ids = batch['prompt_input_ids']
            prompt_attention_mask = batch['prompt_attention_mask']
            with torch.no_grad():
                print("PROMPTS: ", batch['prompt']) 
                outs = self.model.generate(prompt_input_ids, attention_mask=prompt_attention_mask, max_new_tokens=50, num_return_sequences=1, do_sample=True, temperature=0.7, pad_token_id=self.tokenizer.eos_token_id )
                decoded = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
                print("GENERATED: ", list(map(lambda y: y.split('### Argument: ')[-1].strip(), decoded)))
        
                if self.ref_model is None:
                    with self.null_ref_context():
                        outs_ref = self.model.generate(prompt_input_ids, attention_mask=prompt_attention_mask, max_new_tokens=100, num_return_sequences=1, do_sample=True, temperature=0.7, pad_token_id=self.tokenizer.eos_token_id, )        
                else: 
                    outs_ref = self.ref_model.generate(prompt_input_ids, attention_mask=prompt_attention_mask, max_new_tokens=100, num_return_sequences=1, do_sample=True, temperature=0.7, pad_token_id=self.tokenizer.eos_token_id, )
                decoded_ref = self.tokenizer.batch_decode(outs_ref, skip_special_tokens=True)
                print("GENERATED RESPONSES REF: ", list(map(lambda y: y.split('### Argument: ')[-1].strip(), decoded_ref)))

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        self.print_inference(batch)

        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]
        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            output_hidden_states=True,
            **model_kwargs,
        )

        all_logits = outputs.logits
        hidden_states = outputs.hidden_states
        all_logps, _ = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            #average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=False,
            label_pad_token_id=self.label_pad_token_id,
        )
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, hidden_states
    

    def get_batch_loss_metrics(self, model, batch: Dict[str, Union[List, torch.LongTensor]], train_eval: Literal["train", "eval"] = "train"):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        policy_chosen_logps,policy_rejected_logps, policy_chosen_logits, policy_rejected_logits, policy_hidden_states = self.concatenated_forward(model, batch)
        
        # if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
        #     reference_chosen_logps = batch["reference_chosen_logps"]
        #     reference_rejected_logps = batch["reference_rejected_logps"]
        # else:
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    reference_chosen_logps, reference_rejected_logps, _, _, ref_hidden_states = self.concatenated_forward(self.model, batch)
            else:
                reference_chosen_logps, reference_rejected_logps, _, _ , ref_hidden_states= self.concatenated_forward(self.ref_model, batch)

        policy_hidden_states, ref_hidden_states = policy_hidden_states[-1], ref_hidden_states[-1]
        policy_last_hidden_states = policy_hidden_states[:, -1, :]
        ref_last_hidden_states = ref_hidden_states[:, -1, :]

        losses, chosen_rewards, rejected_rewards, clf_loss, chosen_preds, rejected_preds, chosen_loss, rejected_loss = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            batch=batch,
            policy_last_hidden_states=policy_last_hidden_states,
            ref_last_hidden_states=ref_last_hidden_states,
            # fallacy_clf=self.fallacy_clf,
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

        chosen_accuracy = (chosen_preds == 0).float().detach().mean()   
        rejected_accuracy = (rejected_preds == torch.tensor(batch['fallacy_type']).to('cuda')).float().detach().mean()
    
        metrics[f"{prefix}clf_loss"] = clf_loss.detach().mean().cpu()
        metrics[f"{prefix}clf/chosen_loss"] = chosen_loss.detach().mean().cpu()
        metrics[f"{prefix}clf/rejected_loss"] = rejected_loss.detach().mean().cpu()
        metrics[f"{prefix}clf/chosen_accuracy"] = chosen_accuracy.mean().cpu()
        metrics[f"{prefix}clf/rejected_accuracy"] = rejected_accuracy.mean().cpu()
        return losses.mean(), metrics

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        fallacy_clf: nn.Module = None,
        batch=None,
        policy_last_hidden_states=None,
        ref_last_hidden_states=None,
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
        if policy_last_hidden_states is not None:
        
            clf_logits_policy = self.model.classification_head(policy_last_hidden_states)
            #clf_logits_ref = self.ref_model.classification_head(ref_last_hidden_states).to('cuda')
            concatenated_length = clf_logits_policy.size(0)
            
            chosen_logits = clf_logits_policy[:concatenated_length//2]
            rejected_logits = clf_logits_policy[concatenated_length//2:]
            
            chosen_targets = torch.zeros(chosen_logits.size(0), dtype=torch.long).to('cuda')
            rejected_targets = torch.tensor(batch['fallacy_type']).to('cuda')

            class_weights = [0.03, 0.15, 0.09, 0.07, 0.07, 0.03, 0.1, 0.07, 0.11, 0.07, 0.07, 0.06, 0.06, 0.06]
            weights = torch.tensor(class_weights, dtype=torch.float).to('cuda')
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            chosen_loss = loss_fct(chosen_logits, chosen_targets)
            rejected_loss = loss_fct(rejected_logits, rejected_targets)
            clf_loss = chosen_loss + rejected_loss
            losses = losses + self.lambda_*clf_loss
            
            chosen_probabilities = torch.softmax(chosen_logits, dim=1)  ## batch size x num_classes
            rejected_probabilities = torch.softmax(rejected_logits, dim=1) ## batch size x num_classes

        ## if the model is confident that the chosen score is a fallacy -- e.g., sigmoid > 0.5, then the reward should be small for the chosen sample - hence 1-sigmoid
        chosen_rewards = (
            (self.beta ) ## multiply the chosen rewards by sigmoid of the chosen logits after feeding them through the classifier. This way, we can give more weight to the chosen rewards that the classifier is more confident about.
            * (policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device))
            .detach()
        ) 
        rejected_rewards = (
            (self.beta ### multiply the rejected rewards by 1 - sigmoid of the rejected logits after feeding them through the classifier. This way, we can give more weight to the rejected rewards that the classifier is confident about.
            * (policy_rejected_logps.to(self.accelerator.device) - reference_rejected_logps.to(self.accelerator.device))
            .detach()
       ))
        
        if self.current_train_steps % 5 == 0:
            print("CHOSEN preds: ", torch.argmax(chosen_probabilities, dim=1))
            print("-"*50)
            print("TRUE LABELS: ", rejected_targets)
            print("REJECTED preds: ", torch.argmax(rejected_probabilities, dim=1))
            print("-"*50)
        return losses, chosen_rewards, rejected_rewards, clf_loss, chosen_probabilities.argmax(-1), rejected_probabilities.argmax(-1), chosen_loss, rejected_loss

    
    
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        
        self.current_train_steps += 1
        model.train()
        inputs = self._prepare_inputs(inputs)
        self.optimizer2.zero_grad()
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)
        self.optimizer2.step()
        return loss.detach() / self.args.gradient_accumulation_steps
    
    
