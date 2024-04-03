# DONE #python src/ORPO/orpo.py --model-name=mistralai/Mistral-7B-Instruct-v0.2 --beta=0.1 --n-epochs=3 

# DONE #python src/PPO/ppo.py --model-name=mistralai/Mistral-7B-Instruct-v0.2 --ref-model-path=models/arguments/sft_Mistral-7B-Instruct-v0.2_trl_2024-03-12_20:44:21.604653/ --reward-model-path=models/fallacy_clf/howey_electra-large-mnli/ --n-epochs=3 --max-length=128 --batch-size=4

# python src/ORPO/eval_with_gpt.py --orpo-path=models/arguments/orpo_Llama-2-7b-hf_2024-03-26_14:35:13.547014 --model-name=llama
# python src/ORPO/eval_with_gpt.py --orpo-path=... --model-name=mistral

# python src/PPO/eval_with_gpt.py --ppo-path=models/arguments/ppo__2024-04-01_13:10:18.901885 --model-name=llama
# python src/PPO/eval_with_gpt.py --ppo-path=... --model-name=mistral
python src/EVAL/generate_from_models.py --sft-path=models/arguments/sft_mistral/ --dpo-path=models/arguments/dpo_mistral/ --orpo-path=models/arguments/orpo_mistral/ --ppo-path=models/arguments/ppo_mistral/ --model-name=mistral 
python src/ORPO/eval_with_gpt.py --orpo-path=models/arguments/orpo_mistral --model-name=mistral --eval-from-file
python src/PPO/eval_with_gpt.py --ppo-path=models/arguments/ppo_mistral --model-name=mistral --eval-from-file
python src/DPO/eval_with_gpt.py --sft-path=models/arguments/sft_mistral --dpo-path=models/arguments/dpo_mistral --model-name=mistral --task=arguments --eval-from-file
#python src/EVAL/win_rate.py --model-name=llama --sft-path=models/arguments/sft_Llama-2-7b-hf_trl_2024-03-12_09:54:19.188918 --dpo-path=models/arguments/dpo_Llama-2-7b-hf_trl_2024-03-15_09:56:38.508752 --orpo-path=models/arguments/orpo_Llama-2-7b-hf_2024-03-26_14:35:13.547014 --ppo-path=models/arguments/ppo__2024-04-01_13:10:18.901885

