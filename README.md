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

## Preference Optimization



## Training the Binary Classifier: 
```
  python src/fallacy_classifier/train.py --language-model howey/electra-base-mnli --epochs 5 --batch-size 8 --val-batch-size 8 --lr 0.0001 --data-dir data/dpo/arguments --gradient-accumulation 4
```

running the above saves at `models/fallacy_clf/<chosen language model>`

## Running SFT
```
  python src/DPO/sft.py --model-name=google/flan-t5-base --data-dir=data/dpo/arguments --n-epochs=10 --batch-size=16 --output-dir=models/ --is-encoder-decoder=true --use-peft=false
```

This will save the final SFT model in the `models/` directory.

## Running DPO 
<pre>python src/DPO/dpo.py --model-name=google/flan-t5-base <b>--ref-model-path=PATH TO SFT Model</b> --data-dir=data/dpo/arguments <b>--beta=0.5</b> --n-epochs=10 --output-dir=models/ --is-encoder-decoder=true --use-peft=false</pre>

Where bold arguments are new parameters to run DPO. This will also save at the `models/` directory.

## Running Self Reward 
```
  python src/DPO/dpo_self_reward.py --dpo-model-path=<PATH TO DPO Model> --clf-path=<PATH TO Classifier> --fallacy-threshold=0.1
```
