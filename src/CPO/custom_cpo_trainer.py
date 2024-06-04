from trl import CPOTrainer
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch 
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
import torch.optim as optim
from  transformers.utils import is_sagemaker_mp_enabled
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

# Inside CustomDPO class __init__ method


criterion = nn.CrossEntropyLoss()


class CustomCPO(CPOTrainer):
    def __init__(self, custom_eval_steps, *args, **kwargs):
        super(CustomCPO, self).__init__(*args, **kwargs)
        self.optimizer2 = optim.Adam(self.model.classification_head.parameters(), lr=5e-4)
        self.custom_eval_steps = custom_eval_steps
        self.current_train_steps = 0

    def print_inference(self, batch):
        if self.current_train_steps % self.custom_eval_steps == 0:
            prompt_input_ids = batch['prompt_input_ids']
            prompt_attention_mask = batch['prompt_attention_mask']
            with torch.no_grad():
                print("PROMPTS: ", batch['prompt']) 
                outs = self.model.generate(prompt_input_ids, attention_mask=prompt_attention_mask, max_new_tokens=50, num_return_sequences=1, do_sample=True, temperature=0.7)
                decoded = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
                print("GENERATED: ", list(map(lambda y: y.split('### Argument: ')[-1].strip(), decoded)))
        
    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """

        self.print_inference(batch)

        concatenated_batch = self.concatenated_inputs(batch, is_encoder_decoder=self.is_encoder_decoder, label_pad_token_id=self.label_pad_token_id, padding_value=self.padding_value, device=self.accelerator.device)
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs =  {}

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            output_hidden_states=True,
            **model_kwargs,
        )
        all_logits = outputs.logits
        hidden_states = outputs.hidden_states

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                # Shift so that tokens < n predict n
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, hidden_states)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the CPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits, policy_nll_loss, hidden_states = self.concatenated_forward(model, batch)

        policy_last_hidden_state = hidden_states[-1]
        policy_last_hidden_state = policy_last_hidden_state[:, -1, :]

        losses, chosen_rewards, rejected_rewards, clf_chosen_scores, clf_rejected_scores, clf_loss = self.cpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_last_hidden_state=policy_last_hidden_state,
            batch=batch,
        )

        loss = losses.mean() + policy_nll_loss
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
        metrics[f"{prefix}nll_loss"] = policy_nll_loss.detach().mean().cpu()
        metrics['Clf/loss'] = clf_loss.detach().mean().cpu()
        metrics['Clf/chosen_avg_prob'] = clf_chosen_scores.mean().cpu()
        metrics['Clf/rejected_avg_prob'] = clf_rejected_scores.mean().cpu()
        return loss, metrics



    def cpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_last_hidden_state: torch.FloatTensor = None,
        batch: Dict[str, Union[List, torch.LongTensor]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the CPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the CPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        logits = (policy_chosen_logps - policy_rejected_logps).to(self.accelerator.device)

        # The beta is a temperature parameter for the CPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative CPO loss.
        if self.loss_type == "sigmoid":
            # This reduces to Equation 3 from the CPO paper when label_smoothing -> 0.
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

        chosen_rewards = self.beta * (policy_chosen_logps.to(self.accelerator.device)).detach()
        rejected_rewards = self.beta * (policy_rejected_logps.to(self.accelerator.device)).detach()

        
        if policy_last_hidden_state is not None:
        
            clf_logits_policy = self.model.classification_head(policy_last_hidden_state)
            concatenated_length = clf_logits_policy.size(0)
            
            chosen_logits = clf_logits_policy[:concatenated_length//2]
            rejected_logits = clf_logits_policy[concatenated_length//2:]
            
            chosen_targets = torch.zeros(chosen_logits.size(0), dtype=torch.long).to('cuda')
            rejected_targets = torch.tensor(batch['fallacy_type'], dtype=torch.long).to('cuda')

            class_weights = [0.05, 0.15, 0.09, 0.07, 0.07, 0.03, 0.1, 0.07, 0.11, 0.07, 0.07, 0.06, 0.06, 0.06]
            loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to('cuda'))
            chosen_loss = loss_fct(chosen_logits, chosen_targets)
            rejected_loss = loss_fct(rejected_logits, rejected_targets)
            clf_loss = chosen_loss + rejected_loss
            losses = losses + 0.3*clf_loss

            chosen_probabilities = torch.softmax(chosen_logits, dim=1)  ## batch size x num_classes
            rejected_probabilities = torch.softmax(rejected_logits, dim=1)  ## batch size x num_classes
            print("CHOSEN preds: ", torch.argmax(chosen_probabilities, dim=1))
            print("-"*50)
            print("TRUE LABELS: ", batch['fallacy_type'])
            print("REJECTED preds: ", torch.argmax(rejected_probabilities, dim=1))
            print("-"*50)
        # ## if the model is confident that the chosen score is a fallacy -- e.g., sigmoid > 0.5, then the reward should be small for the chosen sample - hence 1-sigmoid
       
        return losses, chosen_rewards, rejected_rewards,chosen_probabilities, rejected_probabilities, clf_loss# F.sigmoid(chosen_logits), F.sigmoid(rejected_logits)

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
    
    
