import logging
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    AutoModel,
    get_linear_schedule_with_warmup, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
)

from LoadData import DatasetLoader
import metrics
logger = logging.getLogger(__name__)


class Classifier:
    def __init__(self,
                 output_model_dir,
                 pretrained_model_name_or_path,                 
                 cache_dir='data/pretrained/',
                 do_lower_case=True,
                 data_dir='data/dpo/',
                 fp16=False):
        
      
        self.output_model_dir = output_model_dir
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.cache_dir = cache_dir
        
        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.fp16 = fp16

        # Setup CUDA, GPU & distributed training
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        
      
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                                                     do_lower_case=do_lower_case,
                                                     cache_dir=self.cache_dir)
        
        self.data_loader = DatasetLoader(tokenizer=self.tokenizer)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def train(self, 
              per_gpu_train_batch_size,
              gradient_accumulation_steps,
              num_train_epochs,
              learning_rate,
              weight_decay=0.0,
              warmup_steps=0,
              adam_epsilon=1e-8,
              max_grad_norm=1.0):

        """ Train the model """
        
        train_batch_size = per_gpu_train_batch_size * max(1, self.n_gpu)
        
        train_dataset, _ = self.data_loader.load_dataset(split='train', data_dir=self.data_dir)
        val_dataset, val_labels = self.data_loader.load_dataset(split='dev', data_dir=self.data_dir)
     
        train_sampler = RandomSampler(train_dataset) 
        
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

       
        model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name_or_path, num_labels=2, ignore_mismatched_sizes=True)
        model.to(self.device)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0},
        ]
       
       
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            train_batch_size
            * gradient_accumulation_steps
        )
        logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        tr_loss, logging_loss = 0.0, 0.0
       
        best_acc = 0
        best_f1 = 0

        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=False
        )
        save_steps = 200#len(train_dataset) // (per_gpu_train_batch_size * gradient_accumulation_steps* self.n_gpu)
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}#, 'label_mask': batch[3]}
                
                
                outputs = model(**inputs)
                
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

               
                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if save_steps > 0 and global_step % save_steps == 0:
                    # Log metrics
                        # Only evaluate when single GPU otherwise metrics may not avg well
                        preds = self._predict(eval_dataset=val_dataset,
                                                per_gpu_eval_batch_size=per_gpu_train_batch_size,
                                                model=model,
                                            )
                        accuracy, f1 = metrics.compute(predictions=preds, labels=val_labels)

                        if accuracy > best_acc or f1 > best_f1:
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(self.output_model_dir)
                            self.tokenizer.save_pretrained(self.output_model_dir)
                            print(f"accuracy improved, previous {best_acc}, new one {accuracy}")
                            print(f"f1 improved, previous {best_f1}, new one {f1}")
                            #print('bleu on dev set improved:', bleu, ' saving model to disk.')
                            best_acc = accuracy
                            best_f1 = f1
                        else:
                            print(f"accuracy not improved, best: {best_acc}, this one: {accuracy}")
                            print(f"f1 not improved, best: {best_f1}, this one: {f1}")
                            #print('bleu on dev set impr
        return global_step, tr_loss / global_step

    def predict(self, per_gpu_eval_batch_size):
        test_dataset, labels = self.data_loader.load_dataset(split='test', data_dir=self.data_dir)
        model = AutoModelForSequenceClassification.from_pretrained(self.output_model_dir, num_labels=2, ignore_mismatched_sizes=True)
        return self._predict(eval_dataset=test_dataset,
                             per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                             model=model, 
                             labels=labels
                            ), labels

    def _predict(self,
                 eval_dataset,
                 model,
                 per_gpu_eval_batch_size,
                 labels=None):

        eval_batch_size = per_gpu_eval_batch_size * max(1, self.n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        model.to(self.device)
        # multi-gpu eval
        if self.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", eval_batch_size)
        preds = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            with torch.no_grad():
                if self.n_gpu > 1:
                    outs = model.module(input_ids=batch[0].to(self.device),
                                            attention_mask=batch[1].to(self.device),
                                        )                   
                else:
                    outs = model(input_ids=batch[0].to(self.device),
                                attention_mask=batch[1].to(self.device),
                                )
                    
                logits = outs.logits.to(self.device) 
                
                   
                probs = torch.softmax(logits, dim=1)    
                for proba in probs:
                    preds.append(torch.argmax(proba).cpu().detach().numpy())
                    

        if labels is not None:
            accuracy, f1 = metrics.compute(predictions=preds, labels=labels)
            print(f"accuracy: {accuracy}, f1: {f1}")
        return preds