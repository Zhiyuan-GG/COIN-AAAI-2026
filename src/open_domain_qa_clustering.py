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

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import CrossEncoder

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
                    default=r'Qwen2.5-14B-Instruct',
                    help='model version')
parser.add_argument('--dataset', default='triviaqa')
parser.add_argument('--max-length-of-generation', type=int, default=128)
parser.add_argument('--num-beams', type=int, default=5, help='for the most likely generation')
parser.add_argument('--num-generations-per-prompt', type=int, default=10, help='for sampling')
parser.add_argument('--top-p', type=float, default=0.9, help='for sampling')
parser.add_argument('--temperature', type=float, default=1.0, help='for sampling')
# ----------------------------------------------------------------------------------------------------------------------
# for bi-entailment
parser.add_argument('--infer-model',
                    default=r'E:\cache_model\deberta-large-mnli',
                    help='local path of infer llm')
# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
# model_name for path of saved parsed dataset
model_name = args.generate_model
print('Generative LLM: ', model_name)
model_path = os.path.join(args.cache_model, args.generate_model)
print('Local path: ', model_path)
# mcqa setting
if args.dataset in ['commonsenseqa']:
    args.max_length_of_generation = 1
elif args.dataset in ['triviaqa', 'coqa']:
    args.max_length_of_generation = 36
# run_name for saving experimental record
run_name = os.path.join(args.record_dir,
                        args.dataset,
                        model_name,
                        'num_generations-' + str(args.num_generations_per_prompt),  # for sampling
                        'temperature-' + str(args.temperature),  # for sampling
                        'num_beams-' + str(args.num_beams),  # for most likely generation
                        'max_len_of_generation-' + str(args.max_length_of_generation))
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
# load generation
with open(f'{run_name}/cleaned_generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)

# for entailment inference
infer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.infer_model)
infer_llm = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.infer_model).cuda()

def semantic_cluster(gen_list):
    ans_cluster_idx = [-1] * len(gen_list)
    cur_cluster_idx = 0
    for i in range(len(gen_list)):
        if ans_cluster_idx[i] == -1:
            ans_cluster_idx[i] = cur_cluster_idx
            for j in range(i + 1, len(gen_list)):
                if ans_cluster_idx[j] == -1:
                    input_1 = gen_list[i] + ' [SEP] ' + gen_list[j]
                    input_2 = gen_list[j] + ' [SEP] ' + gen_list[i]
                    inputs = [input_1, input_2]
                    encoded_input = infer_tokenizer.batch_encode_plus(inputs, padding=True)
                    prediction = infer_llm(torch.tensor(encoded_input['input_ids']).cuda())['logits']
                    prob_vecter = torch.argmax(prediction, dim=1).detach().to('cpu').tolist()

                    if 0 not in prob_vecter and 2 in prob_vecter:
                        ans_cluster_idx[j] = cur_cluster_idx
            cur_cluster_idx += 1
    return ans_cluster_idx

id_to_semantic_cluster_idx = {}
for generation in tqdm.tqdm(generations):
    id = generation['id']
    question = generation['question']
    sampled_generated_texts = generation['sampled_generated_texts']
    qa_list = [question.strip() + ' ' + gen.strip() for gen in sampled_generated_texts]
    sample_cluster_idx = semantic_cluster(qa_list)
    print(sample_cluster_idx)
    id_to_semantic_cluster_idx[id] = sample_cluster_idx

# save
with open(f'{run_name}/sampled_generations_cluster.pkl', 'wb') as record_file:
    pickle.dump(id_to_semantic_cluster_idx, record_file)
print('Record saved to ', f'{run_name}/sampled_generations_cluster.pkl')