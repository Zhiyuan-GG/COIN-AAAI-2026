import argparse
import json
import random

import numpy as np
import tqdm
import re
import os

import accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import pandas as pd
import datasets
from datasets import Dataset
# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--cache-model', default=r'E:\cache_model',
                    help='local path for saving generative llms downloaded from Hugging Face')
parser.add_argument('--generate-model',
                    default=r'DeepSeek-R1-Distill-Qwen-7B',
                    help='model version')
parser.add_argument('--row-data-path',
                    default=r'../row_data/commonsense_qa',
                    help='local path for saving row dataset') # https://huggingface.co/datasets/tau/commonsense_qa
parser.add_argument('--data-dir',
                    default='../datasets',
                    help='local path for saving parsed dataset')
parser.add_argument('--cache-dir',
                    default='../cache',
                    help='cache model from hugging face')
parser.add_argument('--few-shot-num',
                    default=1,
                    help='for few-shot prompt')
parser.add_argument('--max-num',
                    default=10000,
                    help='for maximum applied samples')
args = parser.parse_args()
# ----------------------------------------------------------------------------------------------------------------------
# model_name for path of saved parsed dataset
model_name = args.generate_model
print('Generative LLM (version): ', model_name)
model_path = os.path.join(args.cache_model, args.generate_model)
print('Local path: ', model_path)
# ----------------------------------------------------------------------------------------------------------------------
# Set seed for recurrence
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Fix torch random seed
torch.manual_seed(seed_value)
# cache path for hf_datasets
os.environ["HF_DATASETS_CACHE"] = args.cache_dir
# set cuda device 0,1
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ----------------------------------------------------------------------------------------------------------------------
generative_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path,
                                                     local_files_only=True,
                                                     # resume_download=True,
                                                     # cache_dir=arg.cache_dir,
                                                     # use_auth_token="your_token",
                                                     # proxies='xxx',
                                                     # trust_remote_code=True,
                                                     use_fast=False)
generative_llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                      local_files_only=True,
                                                      torch_dtype=torch.float16,
                                                      # resume_download=True,
                                                      # cache_dir=arg.cache_dir,
                                                      # use_auth_token="your_token",
                                                      # proxies='xxx',
                                                      # trust_remote_code=True,
                                                      device_map="auto")  # require accelerate
# for input_ids length (allowed by llm)
max_input_ids_length = generative_llm.config.max_position_embeddings
# ----------------------------------------------------------------------------------------------------------------------
def form_options(option_letters, option_texts):
    option_str = ''
    for opt_letter, opt_text in zip(option_letters, option_texts):
        option_str += f'{opt_letter}: {opt_text}\n'
    return option_str

def load_dataset(path):
    df = pd.read_parquet(path)
    dict_list = df.to_dict(orient='records')
    return dict_list

if __name__ == "__main__":
    applied_qa = 0
    for_save_json = []
    # load train data
    train_name = 'train-00000-of-00001.parquet'
    train_path = os.path.join(args.row_data_path, train_name)
    train_dataset = load_dataset(train_path)
    print(f'Number of train samples: {len(train_dataset)}')  # 9741
    # load validation data
    validation_name = 'validation-00000-of-00001.parquet'
    validation_path = os.path.join(args.row_data_path, validation_name)
    validation_dataset = load_dataset(validation_path)
    print(f'Number of validation samples: {len(validation_dataset)}')  # 1221

    train_dataset.extend(validation_dataset)
    # clean data
    temp = []
    for idx, train_data in enumerate(tqdm.tqdm(train_dataset)):
        id = train_data['id']
        question = train_data['question']
        option_letters = train_data['choices']['label']
        option_texts = train_data['choices']['text']
        gt_answer = train_data['answerKey']

        formed_options_str = form_options(option_letters, option_texts)
        prompt = '### User:\n' + question + '\n' + formed_options_str + '### Assistant:\n'
        if prompt.isascii():
            temp.append(train_data)
    combined_dataset = temp
    print(f'Total samples: {len(combined_dataset)}')

    # system prompt
    prefix = '### System:\nMake your best effort and select the correct answer for the following multiple-choice ' \
             'question. For each question, only one choice is correct. Answer should be one among A, B, C, D, E.\n\n'

    # few shot prompt
    few_shot_prompt = ''
    idx_for_few_shot_prompt = random.sample(range(0, len(combined_dataset)), args.few_shot_num)
    for idx, data in enumerate(combined_dataset):
        if idx in idx_for_few_shot_prompt:
            question = data['question']
            option_letters = data['choices']['label']
            option_texts = data['choices']['text']
            gt_answer = data['answerKey']

            formed_options_str = form_options(option_letters, option_texts)

            few_shot_prompt += '### User:\n' + question + \
                               '\n' + formed_options_str + \
                               '### Assistant:\n' + gt_answer + '\n\n'
    few_shot_prompt = prefix + few_shot_prompt

    # parse
    dataset = {}
    dataset['prompt'] = []
    dataset['question'] = []
    dataset['options'] = []
    dataset['answer'] = []
    dataset['id'] = []
    for idx, data in enumerate(tqdm.tqdm(combined_dataset)):
        if idx not in idx_for_few_shot_prompt:
            id = data['id']
            question = data['question']
            option_letters = data['choices']['label']
            option_texts = data['choices']['text']
            gt_answer = data['answerKey']

            formed_options_str = form_options(option_letters, option_texts)
            prompt = few_shot_prompt + '### User:\n' + question + '\n' + formed_options_str + '### Assistant:\n'

            input_ids = generative_tokenizer.encode(prompt)
            if len(input_ids) < max_input_ids_length:
                applied_qa += 1
                # for saving options
                saved_option = []
                for opt_letter, opt_text in zip(option_letters, option_texts):
                    saved_option.append(f'{opt_letter}: {opt_text}')

                dataset['prompt'].append(prompt)
                dataset['question'].append(question)
                dataset['options'].append(saved_option)
                dataset['answer'].append(gt_answer)
                dataset['id'].append(id)

                # for check
                for_save_json.append({})
                for_save_json[-1]['prompt'] = prompt
                for_save_json[-1]['question'] = question
                for_save_json[-1]['options'] = saved_option
                for_save_json[-1]['answer'] = gt_answer
                for_save_json[-1]['id'] = str(id)

                if applied_qa == args.max_num:
                    break
    # ------------------------------------------------------------------------------------------------------------------
    # save
    print('Applied samples: ', applied_qa)
    dataset_df = pd.DataFrame.from_dict(dataset)
    dataset = Dataset.from_pandas(dataset_df)
    dataset.save_to_disk(f'{args.data_dir}/commonsenseqa_{model_name}')

    with open(r'../row_data/commonsense_qa/commonsenseqa.json', 'w') as json_file:
        json.dump(for_save_json, json_file, indent=4)

