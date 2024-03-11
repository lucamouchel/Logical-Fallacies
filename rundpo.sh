python3 -u src/DPO/train.py model=t5-base datasets=[cckg] loss=sft exp_name=cckg_t5-base gradient_accumulation_steps=2 batch_size=8 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=true

python3 -u src/DPO/train.py model=t5-base datasets=[cckg] loss=dpo loss.beta=0.25 model.archive=.cache/root/cckg_t5-base_2024-01-10_20-29-28_872117/LATEST/policy.pt exp_name=cckg_dpo_t5-base gradient_accumulation_steps=2 batch_size=32 eval_batch_size=32 trainer=BasicTrainer sample_during_eval=true


python src/DPO/eval_with_gpt.py --ref-model-dir=models/sft_flan-t5-large_trl --model-dir=models/dpo_flan-t5-large_trl --is-encoder-decoder=true --evaluation-type=sft

python src/DPO/sft_with_trl.py --model-name=google/flan-t5-base --n-epochs=2 --batch-size=8 --gradient-accumulation-steps=2 --output-dir=models/test/sft_flan-t5-base_trl --is-encoder-decoder=true 