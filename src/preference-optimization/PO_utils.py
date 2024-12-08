
import argparse
from trl import (
    SFTConfig, 
    DPOConfig, 
    CPOConfig, 
    KTOConfig,  
    SFTTrainer, 
    DPOTrainer, 
    CPOTrainer, 
    KTOTrainer, 
)
from FIPO.FIPOConfig import FIPOConfig
from FIPO.FIPOTrainer import FIPOTrainer

def parse_args():
    parser = argparse.ArgumentParser() 

    parser.add_argument('--train-using', type=str, required=True, help="training to perform -- either sft or preference optimization (sft, dpo, cpo, kto, fipo, ppo)")
    parser.add_argument('--train-data', type=str, default='data/preference-data/train.json', help='path to json training data - for sft and preference optimization')
    parser.add_argument('--model-name', type=str, default=None, help="huggingface model-id e.g., meta-llama/Llama-2-7b-hf")

    ## Specific to preference optimization
    parser.add_argument('--beta', default=0.1, type=float)  
    parser.add_argument('--ref-model-path', default=None)

    ## Specific to PEFT
    parser.add_argument('--peft-config-r', default=16, type=int)
    parser.add_argument('--peft-config-lora-alpha', default=48, type=float)
    parser.add_argument('--peft-config-lora-dropout', default=0.05, type=float)
    parser.add_argument('--use-peft', type=bool, default=True)
    parser.add_argument('--max-length', default=128, type=int)

    ## specific to KTO
    parser.add_argument('--desirable-weight', default=1.0, type=float)
    parser.add_argument('--undesirable-weight', default=1.0, type=float)

    ## Specific to FIPO
    parser.add_argument('--lambda-value', default=0.3, type=float)
    parser.add_argument('--weighting-scheme', default='frequency', type=str, help='frequency or uniform')


    ## Specific to PPO
    parser.add_argument('--reward-model-path')

    ## Training arguments
    parser.add_argument('--n-epochs', default=4, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--eval-batch-size', default=32, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=2, type=int)
    parser.add_argument('--learning-rate', default=1e-5, type=float)
    parser.add_argument('--warmup-steps', default=100, type=int)
    parser.add_argument('--weight-decay', default=0.05, type=float)
    parser.add_argument('--adam-epsilon', default=1e-8, type=float)
    parser.add_argument('--save-steps', default=500, type=int)
    parser.add_argument('--logging-steps', default=300, type=int)
    parser.add_argument('--output-dir', default='models', type=str)

    parser.add_argument('--uniform-data', action='store_true', help='use uniform data for training')
    return parser.parse_args()

def get_trainer_and_config_cls(method_used):
    if method_used == 'sft':
        config = SFTConfig
        trainer = SFTTrainer
        
    if method_used == 'dpo':
        config = DPOConfig
        trainer = DPOTrainer

    elif method_used == 'cpo':
        config = CPOConfig
        trainer = CPOTrainer

    elif method_used == 'kto':        
        config = KTOConfig
        trainer = KTOTrainer

    elif method_used == 'fipo':
        config = FIPOConfig
        trainer = FIPOTrainer
        
    return config, trainer