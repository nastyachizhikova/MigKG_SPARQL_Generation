from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

from datasets import Dataset

import torch

import json
import argparse
import nltk
nltk.download('punkt')

import pandas as pd
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/employment_test/examples.json')
    parser.add_argument('--hf_cache', default=None)
    parser.add_argument('--base_model', default='google-t5/t5-base')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output_name', default='test_T5.csv')


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device)

    with open(args.dataset) as file:
        test_data = json.load(file)

    def preprocess_function(ds):
        output_texts = []

        for i in range(len(ds['input'])):
            text = 'turn into SPARQL: ' + ds['input'][i]
            output_texts.append(text)

        model_inputs = tokenizer(output_texts, truncation=False).to(device)
        
        return model_inputs

    test_dataset = Dataset.from_dict({'input': [el['input'] for el in test_data]})
    test_tokenized_dataset = test_dataset.map(preprocess_function, batched=True)

    # prepare dataloader
    test_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataloader = torch.utils.data.DataLoader(test_tokenized_dataset, batch_size=1)

    all_predictions = []

    for batch in tqdm(dataloader):
        predictions = model.generate(**batch, min_length=50, max_length=300)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        all_predictions.append(decoded_preds)

    res_df = pd.DataFrame() 
    res_df['question'] = [el['input'] for el in test_data]
    res_df['Llama_pred'] = all_predictions

    res_df.to_csv(args.output_name)


if __name__=='__main__':
    main()