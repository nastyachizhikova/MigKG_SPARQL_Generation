import torch

from transformers import (
    AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    AutoTokenizer
)

from datasets import Dataset

import numpy as np

import json
import argparse
import nltk
nltk.download('punkt')

import evaluate

metric = evaluate.load("rouge")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/employment_train/examples.json')
    parser.add_argument('--hf_cache', default=None)
    parser.add_argument('--base_model', default='google-t5/t5-base')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='./models/test')
    parser.add_argument('--test_dir', type=str, default=None, help='Test file directory path if need to be evaluated directly')


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    def model_init():
        return AutoModelForSeq2SeqLM.from_pretrained(args.base_model,
                                                    cache_dir=args.hf_cache)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)


    # preprocess data and create dataset
    with open(args.dataset) as file:
        data = json.load(file)

    # split data into train and dev 
    splitted_dataset = Dataset.from_dict({'input': [el['input'] for el in data], 'output': [el['SPARQL'][0] for el in data]})
    splitted_dataset = splitted_dataset.train_test_split(test_size=0.2)

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    def preprocess_function(ds, tokenizer):
        output_texts = []

        for i in range(len(ds['input'])):
            text = 'turn into SPARQL: ' + ds['input'][i]
            output_texts.append(text)

        model_inputs = tokenizer(output_texts, truncation=False)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(ds["output"], truncation=False)
            model_inputs["labels"] = labels["input_ids"]
            
        return model_inputs

    tokenized_datasets = splitted_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer})


    # configure training parameters
    training_args = Seq2SeqTrainingArguments(
        args.output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        learning_rate=0.0015,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        report_to="none"
    )

    def compute_metrics(eval_pred, tokenizer):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                        for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                        for label in decoded_labels]
        
        # Compute ROUGE scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                                use_stemmer=False)

        # Extract ROUGE f1 scores
        result = {key: value * 100 for key, value in result.items()}
        
        # Add mean generated length to metrics
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                        for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # train
    trainer.train()
    trainer.save_model(f"models/{args.output_dir}")