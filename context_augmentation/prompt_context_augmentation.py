from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
from tqdm.auto import tqdm
import torch
import argparse
import re
import json


def main():
    tqdm.pandas()
    parser = argparse.ArgumentParser(description='Define experiment arguments')
    parser.add_argument('--transformer', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='define the generative model to prompt')
    parser.add_argument('--cache_dir', type=str, default=None, help='define the directory to load the generative model')
    parser.add_argument('--prompt_dir', type=str, default='ent_class_prompt.txt', help='define the path to the prompt')
    parser.add_argument('--data_path', type=str, default='entities_list.csv', help='what dataset are you using')
    parser.add_argument('--token', type=str, default=None, help='token for llama model')
    parser.add_argument('--temperature', type=float, default=0.7, help='set the temperature for the model' )
    parser.add_argument('--batch_size', type=int, default=1, help='set prediction batch size')
    parser.add_argument('--output_path', type=str, default='entities_list_augmented.csv', help='predictions save path')


    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    def prompt_text(entity, question, prompt):
        input_text = prompt.replace('<ENTITY>', entity).replace('<QUESTION>', question)
        inputs = tokenizer(input_text, return_tensors="pt", return_token_type_ids=False).to(device)

        outputs = model.generate(**inputs, generation_config=generation_config)

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return decoded_preds[0].split('assistant')[-1]


    def extract_ent_types(row, data_type='csv'):
        '''
        The function that inputs the logical form,
        extracts all the entities from it and prompts the LLM
        to provide the augmented context given the original question. 
        Output: a dictionary with entities as keys nd augmented contexts as values.
        '''
        if data_type == 'csv':
            question = row['question']
            lisp = row['pred_split_clean_first']
        
        elif data_type == 'json':
            question, lisp = row[0], row[1]

        lisp_entities = re.findall(r']\s?\)? \[ ([\s\'\-\w]+) \] \)', lisp)

        row_dict = {}

        for entity in lisp_entities:
            dec = prompt_text(entity, question, PROMPT)
            row_dict[entity] = dec

        return row_dict


    with open(args.prompt_dir, 'r', encoding='utf-8') as file:
        PROMPT = file.read()

    tokenizer = AutoTokenizer.from_pretrained(args.transformer)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # load model
    model = AutoModelForCausalLM.from_pretrained(args.transformer, 
                                                cache_dir=args.cache_dir, 
                                                trust_remote_code=True,
                                                quantization_config=bnb_config,
                                                device_map="auto",
                                                token=args.token 
                                                )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    generation_config = GenerationConfig(
        max_new_tokens=100,
        min_new_tokens=10,
        num_beams=4,
        do_sample=True,
        early_stopping=False,
        decoder_start_token_id=0,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.3,
        top_p=0.9,
        top_k=0,
        repetition_penalty=1
    )

    print('MODEL LOADED, START RUNNING')

    # read the data according to its format and run the prompting
    if args.data_path.endswith('.csv'):
        data = pd.read_csv(args.data_path)
        data[f'class_pred'] = data.apply(lambda x: extract_ent_types(x, data_type='csv'), axis=1)
        data.to_csv(args.output_path, sep='\t', encoding='utf-8')

    elif args.data_path.endswith('.json'):
        class_gold = []

        with open(args.data_path, 'r') as file:
            gold = json.load(file)

        for example in tqdm(gold):
            example['ent_type_dict'] = extract_ent_types((example['input'], example['output']), data_type='json')
            class_gold.append(example)
            
            with open(args.output_path, 'w') as file:
                json.dump(class_gold, file)


if __name__ == '__main__':
    main()
