
from cProfile import label
from Trainer import Classifier
import os
import argparse

from LoadData import DatasetLoader
import pandas as pd
from transformers import AutoTokenizer
import metrics

DATA_FOLDER = 'data/eval' 


def evaluate(model_dir, per_gpu_eval_batch_size):
    language_model = model_dir.split("/")[-1].replace("_", "/")
    _classifier = Classifier(output_model_dir=model_dir, 
                            cache_dir=os.path.join(DATA_FOLDER, 'pretrained'),
                            pretrained_model_name_or_path=language_model
                                 )
    
    preds, labels = _classifier.predict(per_gpu_eval_batch_size=per_gpu_eval_batch_size)
    acc, f1 = metrics.compute(preds, labels)
    print(f"Accuracy: {acc}, F1: {f1}")
    
def main():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model-dir', dest='model_dir', required=True,
                        help='the folder/google bucket in which the model will be stored or loaded from.')
    
    parser.add_argument('--per-gpu-eval-batch-size', default=4, type=int)
    args = parser.parse_args()
    
    evaluate(model_dir=args.model_dir, per_gpu_eval_batch_size=args.per_gpu_eval_batch_size)

if __name__ == '__main__':
    main()