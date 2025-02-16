import pandas as pd
import json
import evaluate
import re
import numpy as np
import argparse
from pathlib import Path


def get_lisp_aliases(lisp, aliases):
    '''
    Create the list of all equivalent constructions for the relations we have
    '''
    new_lisps = [lisp]

    rels = re.findall(r'JOIN \( R \[ (?:requirement|services|target_group|issued_by|eligible_for|required_for|services_by|target_group_for|issues|target_group_for_document) \] \)|JOIN \[ (?:requirement|services|target_group|issued_by|eligible_for|required_for|services_by|target_group_for|issues|target_group_for_document) \]', lisp)

    if len(rels) == 1:
        new_lisp = lisp.replace(rels[0], aliases[rels[0]]) 
        new_lisps.append(new_lisp)
    
    elif len(rels) == 2:
        new_lisp = lisp.replace(rels[0], aliases[rels[0]]) 
        new_lisps.append(new_lisp)

        new_lisp = lisp.replace(rels[1], aliases[rels[1]]) 
        new_lisps.append(new_lisp)

        new_lisp = lisp.replace(rels[0], aliases[rels[0]]).replace(rels[1], aliases[rels[1]]) 
        new_lisps.append(new_lisp)

    elif len(rels) > 2:
        print(lisp)
    
    return new_lisps


def update_gold(gold):
    '''Get quivalent logical forms for gold standards according to the inverse pattern
    JOIN ( R [ <rel1> ] ) <--> JOIN [ <rel1_inv> ]
    '''

    inverse_rels = {
    "requirement": "required_for", 
    "services": "services_by",
    "target_group": "target_group_for",
    "issued_by": "issues",
    "eligible_for": "target_group_for_document",

    "required_for": "requirement", 
    "services_by": "services",
    "target_group_for": "target_group",
    "issues": "issued_by",
    "target_group_for_document": "eligible_for"
    }

    aliases = {}
    patt1 = "JOIN ( R [ <rel> ] )"
    patt2 = "JOIN [ <rel> ]"

    for rel, rel_inv in inverse_rels.items():
        aliases[patt1.replace('<rel>', rel)] = patt2.replace('<rel>', rel_inv)
        aliases[patt2.replace('<rel>', rel)] = patt1.replace('<rel>', rel_inv)

        aliases[patt2.replace('<rel>', rel_inv)] = patt1.replace('<rel>', rel)
        aliases[patt1.replace('<rel>', rel_inv)] = patt2.replace('<rel>', rel)

    
    gold_aliases = gold.apply(get_lisp_aliases, args=(aliases,))

    return gold_aliases


def clean_prediction(lisp):
    '''
    Remove redundant parentheses and brackets in case there are any generated
    '''
    left_par_n = lisp.count('(')
    right_par_n = lisp.count(')')

    if lisp.endswith(')"]') or lisp.endswith(")']"):
        lisp = lisp[:-2]

    lisp = lisp.replace('{', '[').replace('}', ']')

    while lisp.endswith(' ) )') and right_par_n > left_par_n:
        lisp = lisp[:-2]
        right_par_n = lisp.count(')')

    return lisp


def split_and_clean_70b_pred(lisp):
    lisps = lisp.split(" ) ) (")
    lisps = [l+" ) )" if l.startswith('(') else '('+l+" ) )" for l in lisps]
    lisps = [clean_prediction(l) for l in lisps]

    return lisps


def calculate_exact_acc(pred, gold):
    acc = 0

    for p, g in zip(pred, gold):
        if p in g:
            print(p, g)
            acc += 1

    accuracy = acc / len(pred)

    return accuracy


bleu = evaluate.load("bleu")

def calculate_bleu(pred, gold):
    
    results = bleu.compute(predictions=pred, references=gold)

    return results


def replace_ents(lisp):
    if type(lisp) == str:
        new_lisp = re.sub(r'\( JOIN \[[\w_\s\'\-]+\] [\w+"\\]+ \)', r'( JOIN [] )', lisp)
        new_lisp = re.sub(r'\[[\w_\s\'\-]+\]', r'[]', new_lisp)
    
    elif type(lisp) == list:
        new_lisp = []
        for l in lisp:
            l = re.sub(r'\( JOIN \[[\w_\s\'\-]+\] [\w+"\\]+ \)', r'( JOIN [] )', l)
            l = re.sub(r'\[[\w_\s\'\-]+\]', r'[]', l)
            new_lisp.append(l)
        
    return new_lisp


def calculate_skeleton_match(pred, gold):
    pred_skeletons = [replace_ents(lisp) for lisp in pred]
    gold_skeletons = [replace_ents(lisp_list) for lisp_list in gold]

    acc = 0

    for p, g in zip(pred_skeletons, gold_skeletons):
        if p in g:
            print(p, g)
            acc += 1

    skeleton_match = acc / len(pred)

    return skeleton_match


def extract_rel_list(lisp):
    if type(lisp) == str:
        rels = re.findall(r'(?:R|JOIN) \[([\w,\s]+)\]', lisp)

    elif type(lisp) == list:
        rels = []
        for l in lisp:
            rels.append(re.findall(r'(?:R|JOIN) \[([\w,\s]+)\]', l))
    
    return rels


def calculate_rel_match(pred, gold):
    pred_rel_list = [extract_rel_list(lisp) for lisp in pred]
    gold_rel_list = [extract_rel_list(lisp_list) for lisp_list in gold]

    acc = 0

    for p, g in zip(pred_rel_list, gold_rel_list):
        if p in g:
            print(p, g)
            acc += 1

    rel_match = acc / len(pred)

    return rel_match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_path', default='data/employment_test/examples.json')
    parser.add_argument('--pred_path', default='predictions/ft_cd_8B.csv')
    parser.add_argument('--split', default='split_1')
    parser.add_argument('--generate_clean', default=False, help='Whether or not to save the file with parsed and cleaned predictions')

    args = parser.parse_args()

    pred = pd.read_csv(args.pred_path)

    with open(args.gold_path) as file:
        gold = json.load(file)

    pred['pred'] = pred.Llama_pred.apply(lambda x: x.split('assistant')[-1])
    pred['gold'] = [q['output'] for q in gold]

    # clean the prediction and add the gold aliases
    pred['gold_aliases'] = update_gold(pred.gold)
    pred['pred_clean'] = pred.pred.apply(clean_prediction)

    gold_results = pred.gold_aliases
    pred_results = pred.pred_clean

    if '70B' in args.pred_path or '8B' in args.pred_path:
        pred['pred_split_clean'] = pred.pred_clean.apply(split_and_clean_70b_pred)
        pred['pred_split_clean_first'] = pred.pred_split_clean.apply(lambda x: x[0])

        pred_results = pred.pred_split_clean_first

    acc = calculate_exact_acc(pred=pred_results, gold=gold_results)
    bleu = calculate_bleu(pred=pred_results, gold=gold_results)
    rel_match = calculate_rel_match(pred=pred_results, gold=gold_results)
    skeleton_match = calculate_skeleton_match(pred=pred_results, gold=gold_results)

    res_path = 'results/' + args.split + '/' + Path(args.pred_path).stem + '.csv'

    if args.generate_clean:
        pred.drop(columns=['Llama_pred'], inplace=True)
        pred.to_csv('predictions/' + args.split + '/' + Path(args.pred_path).stem + '_clean.csv')


    res_dict = {}
    res_dict['ACCURACY'] = acc
    res_dict['BLEU'] = bleu['bleu']
    res_dict['SKELETON MATCH'] = skeleton_match
    res_dict['RELATIONS MATCH'] = rel_match

    pd.DataFrame(res_dict, index=[Path(args.pred_path).stem]).to_csv(res_path)


if __name__ == '__main__':
    main()




