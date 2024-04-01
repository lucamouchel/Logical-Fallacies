import argparse
import os
import csv
import json
import re
import token
import tokenize
import sys


sys.path.append(".")
from Trainer import Classifier
DATA_FOLDER = 'data'


def training(per_gpu_train_batch_size,
             data_dir,
             learning_rate,
             epochs,
             language_model,
             grad_acc,
            ):

    classifier = Classifier(
            output_model_dir=f"models/fallacy_clf/{language_model.replace('/', '_')}",
            cache_dir=os.path.join(DATA_FOLDER, 'pretrained'),
            pretrained_model_name_or_path=language_model,
            data_dir=data_dir,
    )
    classifier.train(per_gpu_train_batch_size=per_gpu_train_batch_size,
                        learning_rate=learning_rate,
                        num_train_epochs=epochs,
                        gradient_accumulation_steps=grad_acc)

def parse_args():
    parser = argparse.ArgumentParser(description='train a binary classifier')

    parser.add_argument('--language-model', default='howey/electra-base-mnli', help='Can be either some huggingface model or a '
                                                                         'path to a model. If the path is in GCS we '
                                                                         'download it first.')
    
    parser.add_argument('--epochs', default=5,help='number of epochs to train')
    parser.add_argument('--batch-size', default=8, help='batch size')
    parser.add_argument('--val-batch-size', default=8, help='validation batch size')
    parser.add_argument('--lr', default=0.0001, help='learning rate')
    parser.add_argument('--data-dir', default='data/dpo/arguments', help='data directory')
    parser.add_argument('--gradient-accumulation', default=4)
    return parser.parse_args()

def main():
    args = parse_args()
    training(per_gpu_train_batch_size=int(args.batch_size),
                epochs=int(args.epochs),
                data_dir=args.data_dir,
                learning_rate=float(args.lr),
                language_model=args.language_model,
                grad_acc=int(args.gradient_accumulation))
    
    
if __name__ == '__main__':
    main()
    

