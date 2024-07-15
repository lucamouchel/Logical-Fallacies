# A Logical Fallacy-Informed Framework for Argument Generation

This repository is the official implementation of the paper entitled "_A Logical Fallacy-Informed Framework for Argument Generation_", by Luca Mouchel, Debjit Paul, Shaobo Cui, Robert West, Antoine Bosselut and Boi Faltings.

The pipeline supports any **causal** and **sequence-to-sequence** models from HuggingFace and consists of the following stages: 
- Data Collection with ChatGPT
- Supervised Fine-Tuning (SFT)
- Preference Optimization with existing methods (DPO, KTO, CPO, PPO) and our method (FIPO)
- _win-rate_ and _fallacy-rate_ evaluations with GPT-4


## Automatically Collecting Data with ChatGPT 
The scripts for this step are in `src/augment_arguments.py`

## Superised Fine-Tuning (SFT)
To implement SFT, you can run 
```bash
python src/preference-optimization/trainer.py --train-using=SFT --train-data=data/sft/train.json --model-name=<HF model-id> --use-peft=True 
```

You can also add extra training arguments which have a default value, including:
```bash
python src/preference-optimization/trainer.py ... --n-epochs=<> --batch-size=<> --gradient-accumulation-steps=<> --learning-rate=<> --warmup-steps=<> --weight-decay=<> --logging-steps=<> --save-steps=<> --output-dir=<> 
```

By default, running SFT will save the trained model in `models/sft_<model-name>` e.g., if you train with Llama-2-7b (`meta-llama/Llama-2-7b-hf`), we will save the model as `models/sft_Llama-2-7b-hf`. 

## Preference Optimization
### Methods Requiring a Reference Model (DPO, KTO and PPO)
For methods requiring a reference model, you don't have to specify the model-id, however, you must provide the path to the reference (SFT) model previously trained.
#### DPO
the required arguments are the following - and you can add extra training arguments as mentioned above
```bash
python src/preference-optimization/trainer.py --train-using=DPO --beta=<> --ref-model-path=<Path to SFT model> --train-data=data/preference-data/train.json
```
#### KTO
For KTO, we have two additional arguments, which both default to `1.0`
```bash 
python src/preference-optimization/trainer.py --train-using=KTO --beta=<> --ref-model-path=<Path to SFT model> --desirable-weight=<> --undesirable-weight=<> --train-data=data/preference-data/train.json
```

#### PPO 
PPO requires explicit reward modelling - for which we need a reward model. 
The reward model we pick is a binary fallacy argument classifier (0 - not a fallacy; 1 - fallacy) which uses the preferred and dispreferred arguments in the preference dataset.
You can train the binary classifier using: 
```bash
python src/fallacy_classifier/train.py --language-model=<> --epochs=<> --batch-size=<> --val-batch-size=<> --lr=<> --data-dir=<> --gradient-accumulation=<> --train-data=data/preference-data/train.json
```
By default, we use the `howey/electra-large-mnli` language model. The classifier is then saved in the folder `models/fallacy_clf/.`
You can then run PPO using
```bash
python src/preference-optimization/trainer.py --train-using=PPO --ref-model-path=<Path to SFT model> --reward-model-path=models/fallacy_clf/<> --train-data=data/preference-data/train.json
```
The rewards used during the PPO phase are the "_not a fallacy_" logits - i.e., 
```python
tokens = reward_tokenizer(generated_responses)
rewards = reward_model(**tokens)[:, 0] ## the model outputs two logits in the form [not a fallacy logit, is a fallacy logit]
```
we use the not a fallacy logits as rewards.

### Reference Free Methods (CPO, FIPO)
These methods are easier to run. Here you should not specify a reference model path, instead, specify the huggingface model-id (e.g., meta-llama/Llama-2-7b-hf)
#### CPO
```bash
python src/preference-optimization/trainer.py --train-using=CPO --model-name=<HF model id> --beta=<> --train-data=data/preference-data/train.json
```

#### FIPO
This is the method we introduce in our paper, it uses CPO as the backbone preference optimization method, and adds a classification loss on top. The loss is defined by

$$\mathcal{L}_{\text{FIPO}} =  \mathcal{L} _ \theta +\lambda\mathcal{L} _ \text{FI} $$

where in our case $\theta$ is CPO, and $\mathcal{L} _ \text{FI}$ is the additional loss, as a weighted cross-entropy loss and $\lambda$ is a weighting parameter.

You can also specify the weighting scheme - either uniform or frequency. The frequency works better, as it gives a larger weight to fallacy types occurring more often - teaching the model to learn more from certain fallacy types, rather than having the same penalty for all types.
To run FIPO:
```bash
python src/preference-optimization/trainer.py --train-using=FIPO --model-name=<HF model id> --lambda-value=<> --weighting-scheme=<frequency or uniform> --beta=<> --train-data=data/preference-data/train.json
```

# Examples 
### DPO with Llama-2
Here is an example of running the pipeline:

1. SFT -
```bash
python src/preference-optimization/trainer.py --train-using=SFT --train-data=data/sft/train.json --model-name=meta-llama/Llama-2-7b-hf --use-peft=True
```
2. DPO -
```bash
python src/preference-optimization/trainer.py --train-using=DPO --train-data=data/preference_optimization/train.json --ref-model-path=models/sft_Llama-2-7b-hf
```
