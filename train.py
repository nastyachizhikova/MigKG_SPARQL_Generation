import transformers

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from datasets import Dataset

from peft import LoraConfig

from trl import SFTTrainer 

import torch

import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/employment_train/examples.json')
    parser.add_argument('--hf_cache', default=None)
    parser.add_argument('--base_model', default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--prompt', default='prompts/default_llama3.txt')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--save_path', default='test_ft')
    parser.add_argument('--token', default=None)


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # load model
    model = AutoModelForCausalLM.from_pretrained(args.base_model, 
                                                cache_dir=args.hf_cache, 
                                                trust_remote_code=True,
                                                quantization_config=bnb_config,
                                                token=args.token,
                                                device_map="auto"
                                                )


    # create dataset
    with open(args.dataset) as file:
        data = json.load(file)

    with open(args.prompt, encoding='utf-8') as file:
        prompt = file.read()

    dataset = Dataset.from_dict({'input': [el['input'] for el in data], 'output': [el['output'] for el in data]})

    # training configuration
    def formatting_func(example):
        output_texts = []
        for i in range(len(example['input'])):
            text = prompt.replace('{{input}}', example['input'][i]).replace('{{output}}', example['output'][i])
            output_texts.append(text)
        return output_texts

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'v_proj'],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            lr_scheduler_type='cosine',
            logging_steps=10,
            save_steps=1000,
            learning_rate=1e-5,
            fp16=True,
            num_train_epochs=args.num_epochs,
            output_dir=f"outputs/{args.save_path}",
            optim="paged_adamw_8bit",
            report_to="none"
        ),
        formatting_func=formatting_func
    )

    trainer.train()
    trainer.save_model(f"models/{args.save_path}")
