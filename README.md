# MigKG SPARQL Generation
This repository contains the code used for the experiments described in a paper on SPARQL generation for a closed domain.

First, make sure your environment has all the necessary packages.

``` bash
conda env create -f sparql_env.yml
```

To fine-tune Llama for Logical Query generation, use the following command:

``` bash
python train.py --dataset data/train/examples.json 
                --base_model meta-llama/Meta-Llama-3-8B-Instruct 
                --checkpoint None 
                --save_path ft_8B 
                --num_epochs 20
                --token your_hf_token_here
```

Note: Make sure you got access to use Llama models from HuggingFace and replace `your_hf_token_here` with your HuggingFace token.

To predict logical forms with the fine-tuned model:

``` bash
python predict.py --dataset data/test/examples.json 
                  --hf_cache ../.hf_home 
                  --base_model meta-llama/Meta-Llama-3-8B-Instruct 
                  --checkpoint models/ft_8B 
                  --prompt prompts/default_llama3.txt 
                  --output_name predictions/ft_8B.csv
                  --token your_hf_token_here

```

To evaluate your generation module, the following command can be used:

``` bash
python evaluate_metrics.py --pred_path predictions/ft_8B.csv 
                           --gold_path data/test/examples.json 
                           --split split_1

```

Next, you want to run retrieval to get the indices of your predicted entities in the given KG. 

``` bash
python retrieve.py --gold_path data/test/examples.json 
                   --pred_path ../predictions/ft_8B.csv --mode prediction

```

If you want to run retrieval with augmented contexts, first run the prompting for augmentation and then run the retrieval.

``` bash
cd context_augmentation
python prompt_context_augmentation.py --transformer meta-llama/Meta-Llama-3-8B-Instruct 
                                      --data_path ../data/test/examples.json --batch_size 2 --prompt_dir prompt_context.txt --output_path ft_8B_augmented.csv

cd ../
python retrieve.py --gold_path data/test/examples.json 
                   --pred_path predictions/ft_8B.csv
                   --augmented_contexts_path context_augmentation/ft_8B_augmented.json 
                    --mode prediction 
                    --results_path ft_8B_retrieved.csv

```

Finally, evaluate your results.

```bash
python evaluate_retrieval.py --gold_path data/test/examples.json 
                   --pred_path predictions/ft_8B.csv
                   --augmented_contexts_path context_augmentation/ft_8B_augmented.json 
                    --mode prediction 
                    --results_path ft_8B_retrieved.csv
```
