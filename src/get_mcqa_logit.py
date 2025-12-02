import argparse
import pathlib
import pickle
import tqdm
import os
import random
import json

import accelerate
import datasets
from collections import Counter

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch.nn.functional as F
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir',
                    default='../datasets',
                    help='save parsed dataset')
parser.add_argument('--cache-dir',
                    default='../cache',
                    help='cache model from hugging face')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
# ----------------------------------------------------------------------------------------------------------------------
# for run name
parser.add_argument('--record-dir',
                    default='../records',
                    help='save experimental records')
parser.add_argument('--cache-model', default=r'E:\cache_model',
                    help='local path of generative llm downloaded from Hugging Face')
parser.add_argument('--generate-model',
                    default=r'vicuna-13b-v1.5',
                    help='model version')
parser.add_argument('--dataset', default='commonsenseqa')  ##
# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
# model_name for path of saved parsed dataset
model_name = args.generate_model
print('Generative LLM: ', model_name)
model_path = os.path.join(args.cache_model, args.generate_model)
print('Local path: ', model_path)
# for save path
run_name = os.path.join(args.record_dir,
                        args.dataset,
                        model_name)
# mcqa setting
if args.dataset in ['commonsenseqa']:
    args.max_length_of_generation = 1
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
# load LLM, tokenizer
generative_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path,
                                                     local_files_only=True,
                                                     # resume_download=True,
                                                     # cache_dir=arg.cache_dir,
                                                     # use_auth_token="your_token",
                                                     # proxies='xxx',
                                                     # trust_remote_code=True,
                                                     use_fast=False)
if 'Qwen' in model_name:
    dtype = torch.bfloat16
else:
    dtype = torch.float16
generative_llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                      local_files_only=True,
                                                      torch_dtype=dtype,
                                                      # resume_download=True,
                                                      # cache_dir=arg.cache_dir,
                                                      # use_auth_token="your_token",
                                                      # proxies='xxx',
                                                      # trust_remote_code=True,
                                                      device_map="auto")  # require accelerate
# ----------------------------------------------------------------------------------------------------------------------
dataset = datasets.load_from_disk(f'{args.data_dir}/{args.dataset}_{model_name}')
id_to_logits = dict()

for data in tqdm.tqdm(dataset):
    id = data['id']
    prompt = data['prompt']
    options = data['options']
    input_ids = generative_tokenizer(prompt, return_tensors="pt")['input_ids'].cuda()

    option_letters = [temp[0] for temp in options]
    choice_ids = [generative_tokenizer.convert_tokens_to_ids(c) for c in option_letters]

    with torch.no_grad():
        outputs = generative_llm(input_ids).logits[0, -1, :]
        choice_logits = outputs[choice_ids]
        probs = F.softmax(choice_logits, dim=0).detach().cpu()

        id_to_logits[id] = probs.tolist()
        print(id_to_logits[id])

# save to records
pathlib.Path(run_name).mkdir(parents=True, exist_ok=True)
# save
with open(f'{run_name}/logit.pkl', 'wb') as record_file:
    pickle.dump(id_to_logits, record_file)
print('Record saved to ', f'{run_name}/logit.pkl')
