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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1', cache_dir='/m/cs/scratch/trust-m/chizhia1/thesis/.hf_home').to(device)


pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer, padding=True, truncation=True, framework="pt", batch_size=16)

def get_embedding(text):  
    '''
    This function inputs a piece of text and returns semantic features
    '''  
    output = pipe(text, return_tensors = "pt")
    # get the cls token embedding
    text_emb = output[0, 0, :].detach().numpy()

    return text_emb



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


def get_averaged_embedding(hdb_labels_embs):
    '''
    The function inputs the results of the hierarchical clusterization
    and outputs an index of entity aliases and the averaged embedding
    '''
    emb_dict = {key: [] for key in list(set([x[1] for x in hdb_labels_embs])) if key != -1}
    ent_dict = {key: [] for key in list(set([x[1] for x in hdb_labels_embs])) if key != -1}

    # differenciate all the outliers with negative cluster labels
    outlier_counter = 0

    for (ent, label, emb) in hdb_labels_embs:
        if label != -1:
            emb_dict[label].append(emb)
            ent_dict[label].append(ent)
        else:
            label += outlier_counter
            emb_dict[label] = [emb]
            ent_dict[label] = [ent]
            outlier_counter += -1

    for key, value in emb_dict.items():
        emb_dict[key] = np.mean(np.stack(value), axis=0)

    assert emb_dict.keys() == ent_dict.keys()

    return emb_dict, ent_dict


def find_closest_entity_in_type(entity_emb, entity_index, embeddings, idx2cluster):
    '''
    This function takes the entity embedding and the index of 
    embeddings from one entity class and returns the closest entity of the class 
    and the cosine distance value
    '''
    dist = cosine_distances(embeddings, entity_emb.reshape(1, -1))
    min_idx = np.argmin(dist)
    min_dist = dist[min_idx]

    cluster_idx = idx2cluster[min_idx]
    entity = entity_index[cluster_idx]

    return entity, cluster_idx, min_dist


def retrieve_entity_from_KG(entity_text, entity_types=None, mode=None):

    if mode == 'context_aug':
        augm_contexts = entity_types
        
        entity_types = [
                'Document', 
                'Organization', 
                'Social Group', 
                'Service', 
                'Activity', 
                'Language', 
                'Contact information', 
                'Description', 
                'Cost', 
                'Location', 
                'Frequency', 
                'Time', 
                'Date', 
                'Country',
                'OTHER'
            ]
        
        ent_emb = get_embedding(entity_text)
        context_emb = get_embedding(augm_contexts[0])
        entity_emb = np.mean([ent_emb, context_emb], axis=0)

    elif mode == 'ent_class':
        entity_emb = get_embedding(entity_text)
    
    else:
        entity_types = [
                'Document', 
                'Organization', 
                'Social Group', 
                'Service', 
                'Activity', 
                'Language', 
                'Contact information', 
                'Description', 
                'Cost', 
                'Location', 
                'Frequency', 
                'Time', 
                'Date', 
                'Country',
                'OTHER'
            ]
        
        entity_emb = get_embedding(entity_text)

    # iterate over all entities in each type and if save the element with the least cosine distance
    ent_min_distance = 1
    retrieved_entity = None
    retrieved_cluster_idx = None
    retrieved_type = None


    for type in entity_types:
        with open(f'../../KG_data/resolution_res/{type}_hbd_labels.txt', 'r', encoding='utf-8') as file:
            entity_index = eval(file.read())

        with open(f'../../KG_data/resolution_res/{type}_embeddings.npy', 'rb') as file:
            embeddings = np.load(file)
        
        with open(f'../../KG_data/resolution_res/{type}_index2cluster.txt', 'r', encoding='utf-8') as file:
            idx2cluster = eval(file.read())

        entity, cluster_idx, min_dist = find_closest_entity_in_type(entity_emb, entity_index, embeddings, idx2cluster)
       
        if min_dist < ent_min_distance:
            ent_min_distance = min_dist
            retrieved_entity = entity
            retrieved_cluster_idx = cluster_idx
            retrieved_type = type

    class2idx = {
        'Document': 'doc', 
        'Organization': 'org', 
        'Social Group': 'soc', 
        'Service': 'ser', 
        'Activity': 'act', 
        'Language': 'lang', 
        'Contact information': 'cont', 
        'Description': 'desc', 
        'Cost': 'cost', 
        'Location': 'loc', 
        'Frequency': 'freq', 
        'Time': 't', 
        'Date': 'd', 
        'Country': 'c',
        'OTHER': 'other'
    }
    
    ent_kg_idx = class2idx[retrieved_type]+ '_' + str(retrieved_cluster_idx)

    return ent_kg_idx, retrieved_entity

def lisp_to_idx(lisp, ent_types_dict=None, mode=None):
    entities = re.findall(r']\s?\)? \[ ([\s\'\-\w]+) \] \)', lisp)
    ent_indices = []

    for ent in entities:
        if mode == 'context_aug':
            if ent in ent_types_dict:
                ent_type = [ent_types_dict[ent]]
            elif mode == 'context_aug':
                ent_type = [ent]
        else:
            ent_type = None

        ent_kg_idx, retrieved_entity = retrieve_entity_from_KG(ent, ent_type, mode=mode)
        ent_indices.append(ent_kg_idx)

    for ent, ent_idx in zip(entities, ent_indices):
        lisp = lisp.replace(ent, ent_idx)
    
    return lisp


def clean_lisp(lisp):
    '''
    Clean the S-expression so that it is parsable by the convertion function
    '''
    return lisp.replace('( ', '(') \
                .replace(' )', ')') \
                .replace('[ ', '') \
                .replace(' ]', '') \
                .replace("'", "")


def evaluate_retrieval(pred_path, gold_path, augmented_contexts_path=None, mode='gs_annotations', results_path='res.csv'):
    if mode == 'predictions':
        pred_df = pd.read_csv(pred_path, sep=',')
        pred_lisp = pred_df['pred_split_clean_first']
        questions = pred_df['question']

        if augmented_contexts_path:

            # conduct retrieval using context augmentation
            with open(augmented_contexts_path, 'r') as file:
                class_pred = json.load(file)
                ent_contexts = [out['class_pred'] for out in class_pred]

            pred_lisp_indexed = [lisp_to_idx(l, ent_contexts_dict, mode='context_aug') for l, ent_contexts_dict in zip(pred_lisp, ent_contexts)]
        
        else:
            pred_lisp_indexed = [lisp_to_idx(l) for l in tqdm(pred_lisp)]
    
    elif mode == 'gs_annotations':

        with open(gold_path, 'r') as file:
            gold = json.load(file)
            pred_lisp = [out['output'] for out in gold]

        if augmented_contexts_path:

            # conduct retrieval using context augmentation
            with open(augmented_contexts_path, 'r') as file:
                class_pred = json.load(file)
                ent_contexts = [out['ent_type_dict'] for out in class_pred]

            pred_lisp_indexed = [lisp_to_idx(l, ent_contexts_dict, mode='context_aug') for l, ent_contexts_dict in zip(pred_lisp, ent_contexts)]
        
        else:
            pred_lisp_indexed = [lisp_to_idx(l) for l in pred_lisp]

    pred_sparql = []

    for l in pred_lisp_indexed:
        try:
            pred_sparql.append(lisp_to_sparql(clean_lisp(l)))
        except:
            pred_sparql.append("")

    retr_pred_df = pd.DataFrame()
    retr_pred_df['pred'] = pred_sparql

    retr_pred_df.to_csv(results_path)
    

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, default=None, help='the path to predicted logical forms')
    parser.add_argument('--gold_path', type=str, default=None, help='the path to gold standard logical forms')
    parser.add_argument('--augmented_contexts_path', type=str, default=None, help='the path to the file with augmented contexts')
    parser.add_argument('--mode', type=str, 
                                  default='gs_annotations', 
                                  choices=['gs_annotations', 'predictions'],
                                  help='whether we run the retrieval over the gold standard \
                                    textual logical forms or over model-generated predictions')
    parser.add_argument('--results_path', type=str, default=None, help='the path to how to save prediction results')
                
    args = parser.parse_args()

    evaluate_retrieval(pred_path=args.pred_path, 
                       gold_path=args.gold_path, 
                       augmented_contexts_path=args.augmented_contexts_path,
                       mode=args.mode,
                       context_aug=args.context_aug,
                       results_path=args.results_path
                       )


if __name__ == '__main__':
    main()





