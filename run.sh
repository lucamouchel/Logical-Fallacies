python src/CPO/cpo.py --task=arguments --model-name=mistralai/Mistral-7B-Instruct-v0.2 --n-epochs=3 --max-length=128 
python src/KTO/kto.py --task=arguments --ref-model-path=models/arguments/sft_mistral  --model-name=mistralai/Mistral-7B-Instruct-v0.2 --n-epochs=3 --max-length=128 


python src/EVAL/generate_phrases.py --model-path=models/arguments/cpo_mistral --model-name=mistral --type=cpo
python src/EVAL/generate_phrases.py --model-path=models/arguments/kto_mistral --model-name=mistral --type=kto

python src/EVAL/win_rate.py --model-name=mistral 

python src/KTO/eval_with_gpt.py --kto-path=models/arguments/kto_mistral --model-name=mistral --eval-from-file
python src/CPO/eval_with_gpt.py --cpo-path=models/arguments/cpo_mistral --model-name=mistral --eval-from-file

