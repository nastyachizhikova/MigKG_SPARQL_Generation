from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from transformers.generation import GenerationConfig

from peft import PeftModel

import torch
import argparse
import pandas as pd
from tqdm.auto import tqdm

import json


def get_input_text(input_question, prompt_template=None):
    prompt = prompt_template.replace('{{input}}', input_question)
    prompt = prompt.replace('{{output}}', '')

    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/employment_test/examples.json')
    parser.add_argument('--base_model', default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--prompt', default='prompts/default_llama3.txt')
    parser.add_argument('--output_name', default='test.csv')
    parser.add_argument('--token', default=None)
    parser.add_argument('--hf_cache', default=None)


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # load model
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, 
                                                    cache_dir=args.hf_cache, 
                                                    trust_remote_code=True,
                                                    quantization_config=bnb_config,
                                                    device_map="auto",
                                                    token=args.token, 
                                                )

    if args.checkpoint != None:
        model = PeftModel.from_pretrained(base_model, 
                                        args.checkpoint,  
                                        device_map="auto", 
                                        is_trainable=False)
    else:
        model = base_model

    tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            use_fast=True,
            padding_side="right", # training with left-padded tensors in fp16 precision may cause overflow
            token=args.token
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    # load test data
    if args.dataset.endswith('.csv'):
        test_df = pd.read_csv(args.dataset)

        with open(args.prompt, 'r', encoding='utf-8') as file:
            prompt_template = file.read()

        questions = test_df.question
        input_texts = questions.apply(get_input_text, prompt_template=prompt_template)

    elif args.dataset.endswith('.json'):
        with open(args.dataset, 'r') as file:
            test_data = json.load(file)
        
        with open(args.prompt, 'r', encoding='utf-8') as file:
            prompt_template = file.read()
        
        questions = [question['input'] for question in test_data]
        input_texts = [get_input_text(q, prompt_template=prompt_template) for q in questions]


    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.do_sample = True
    gen_cfg.min_new_tokens = 10
    gen_cfg.max_new_tokens = 100
    gen_cfg.temperature = 0.95
    gen_cfg.top_p = 0.7

    # generate predictions
    predictions = []

    for input_text in tqdm(input_texts):
        print(input_text)
        inputs = tokenizer(input_text, padding=True, return_tensors="pt", return_token_type_ids=False).to(device)

        outputs = model.generate(**inputs, generation_config=gen_cfg)
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.append(decoded_preds)

    res_df = pd.DataFrame() 
    res_df['question'] = questions
    res_df['Llama_pred'] = predictions

    res_df.to_csv(args.output_name)


if __name__ == '__main__':
    main()
