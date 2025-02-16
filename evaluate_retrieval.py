import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoTokenizer, AutoModel, pipeline
from logic_form_util import lisp_to_sparql
import json
import re
from tqdm.auto import tqdm
import argparse
from pathlib import Path
import evaluate
import random

random.seed(0)


def calculate_exact_acc(pred, gold):
    acc = 0

    for p, g in zip(pred, gold):
        if p in g:
            acc += 1

    accuracy = acc / len(pred)

    return accuracy


bleu = evaluate.load("bleu")

def calculate_bleu(pred, gold):
    
    results = bleu.compute(predictions=pred, references=gold)

    return results

metric = evaluate.load("rouge")

def calculate_rouge(predictions, gold):    

    # Compute ROUGE scores
    result = metric.compute(predictions=predictions, references=gold,
                            use_stemmer=False)

    # Extract ROUGE f1 scores
    result = {key: value * 100 for key, value in result.items()}
        
    return {k: round(v, 4) for k, v in result.items()}


def clean_lisp(lisp):
    '''
    Clean the S-expression so that it is parsable by the convertion function
    '''
    return lisp.replace('( ', '(') \
                .replace(' )', ')') \
                .replace('[ ', '') \
                .replace(' ]', '') \
                .replace("'", "")


def evaluate_retrieval(pred_path, gold_path, res_path):
    pred_sparql = []
    pred_df = pd.read_csv(pred_path, sep=',')

    if 'T5' in pred_path:
        pred_sparql = [pred.strip('[]').strip('""') for pred in pred_df['Llama_pred']]
    elif 'zero_shot' in pred_path:
        pred_sparql = [pred.split('}assistant')[1] for pred in pred_df['Llama_pred']]
    else:
        pred_sparql = [pred for pred in pred_df['pred']]


    with open(gold_path, 'r') as file:
        gold = json.load(file)

    gold_sparql = [out['SPARQL'] for out in gold]

    res_dict = {}
    res_dict['ACCURACY'] = str(calculate_exact_acc(pred_sparql, gold_sparql))
    res_dict['BLEU'] = str(calculate_bleu(pred_sparql, gold_sparql))
    res_dict['ROUGE'] = str(calculate_rouge(pred_sparql, gold_sparql))

    pd.DataFrame(res_dict, index=[Path(pred_path).stem]).to_csv(res_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, default=None)
    parser.add_argument('--gold_path', type=str, default=None)
    parser.add_argument('--res_path', type=str, default='results.csv', help='path to how to save the results df')
    
    
    args = parser.parse_args()

    evaluate_retrieval(pred_path=args.pred_path, 
                       gold_path=args.gold_path,
                       res_path=args.res_path
                       )


if __name__ == '__main__':
    main()





