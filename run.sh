python src/DPO/dpo.py --ref-model-path=models/arguments/sft_mistral --beta=0.1 --n-epochs=3 --max-length=150 --lambda-value=0.01 --batch-size=4 --lambda-value=0.3 --output-dir=models_BIS
python src/CPO/cpo.py --model-name=mistralai/Mistral-7B-Instruct-v0.2 --n-epochs=2 --output-dir=models_BIS 
