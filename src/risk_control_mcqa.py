import argparse
import math
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
import scipy.stats as st
from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt

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
parser.add_argument('--dataset', default='commonsenseqa')
parser.add_argument('--max-length-of-generation', type=int, default=128)
parser.add_argument('--num-beams', type=int, default=5, help='for the most likely generation')
parser.add_argument('--num-generations-per-prompt', type=int, default=20, help='for sampling')
parser.add_argument('--top-p', type=float, default=0.9, help='for sampling')
parser.add_argument('--temperature', type=float, default=1.0, help='for sampling')
# ----------------------------------------------------------------------------------------------------------------------
# for risk control
parser.add_argument('--apply', type=int, default=6000, help='applied number')
parser.add_argument('--uncertainty', type=str, default='white', help='white or black')
parser.add_argument('--split-ratio', type=float, default=0.5, help='for calib and test num')
parser.add_argument('--multi-check', type=int, default=100, help='for multiple split and check')
# parser.add_argument('--alpha', type=float, default=0.1, help='risk level')
parser.add_argument('--delta', type=float, default=0.05, help='significance level')
parser.add_argument('--upper-bound', type=str, default='CP', help='CP (Clopper–Pearson) OR HFD (Hoeffding)')
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
run_name_generation = os.path.join(args.record_dir,
                                   args.dataset,
                                   model_name,
                                   'num_generations-' + str(args.num_generations_per_prompt),  # for sampling
                                   'temperature-' + str(args.temperature),  # for sampling
                                   'num_beams-' + str(args.num_beams),  # for most likely generation
                                   'max_len_of_generation-' + str(args.max_length_of_generation))
run_name_logit = os.path.join(args.record_dir,
                              args.dataset,
                              model_name)
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
# load most_likely_generated_text and sampled_generated_texts
with open(f'{run_name_generation}/cleaned_generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)
generations = generations[:args.apply]
# load logit
with open(f'{run_name_logit}/logit.pkl', 'rb') as record_file:
    id_to_logit = pickle.load(record_file)
# ----------------------------------------------------------------------------------------------------------------------
# uncertainty
id_to_uncertainty = {}
id_to_correctness = {}
for gen in tqdm.tqdm(generations):
    id = gen['id']
    gt_answer = gen['answer']
    most_likely_generated_text = gen['most_likely_generated_text']
    sampled_generated_texts = gen['sampled_generated_texts']
    option_letters = [temp[0] for temp in gen['options']]
    logit = id_to_logit[id]
    # correctness
    correctness = int(gt_answer == most_likely_generated_text)
    id_to_correctness[id] = correctness
    # uncertainty
    if args.uncertainty == 'white':
        # for probability (white-box)
        response_probability_dict = {option: logit for logit, option in zip(logit, option_letters)}
        # predictive entropy white/black
        probabilities_white = np.array(list(response_probability_dict.values()))
        predictive_entropy = -np.sum(
            probabilities_white[probabilities_white > 0] * np.log(probabilities_white[probabilities_white > 0]))
    elif args.uncertainty == 'black':
        # for frequency (black-box)
        response_counter_dict = Counter(sampled_generated_texts)
        response_frequency_dict = {option: response_counter_dict[option] / len(sampled_generated_texts) for option in
                                   option_letters}
        probabilities_black = np.array(list(response_frequency_dict.values()))
        predictive_entropy = -np.sum(
            probabilities_black[probabilities_black > 0] * np.log(probabilities_black[probabilities_black > 0]))
    id_to_uncertainty[id] = predictive_entropy
# ----------------------------------------------------------------------------------------------------------------------
qa_num = len(generations)
num_calib = int(qa_num * args.split_ratio)
num_test = qa_num - num_calib
alpha_list = [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45, 0.5]
mean_list = []
std_list = []
for alpha in alpha_list:
    multi_check_emr_list = []
    for epoch in tqdm.tqdm(range(args.multi_check)):
        # for reproduction
        random.seed(epoch)
        calib_set = random.sample(generations, num_calib)
        test_set = [gen for gen in generations if gen not in calib_set]
        # --------------------------------------------------------------------------------------------------------------
        # sorted_uncertainties = sorted(set([id_to_uncertainty[d['id']] for d in calib_set]))
        min_uncertainty = min([id_to_uncertainty[d['id']] for d in calib_set])
        max_uncertainty = max([id_to_uncertainty[d['id']] for d in calib_set])
        sorted_uncertainties = np.linspace(min_uncertainty, max_uncertainty, num_calib).tolist()
        low = 0
        high = len(sorted_uncertainties) - 1
        t_hat = None
        while low <= high:
            mid = (low + high) // 2
            t_candidate = sorted_uncertainties[mid]
            m_cal = 0
            w_cal = 0
            for calib_data in calib_set:
                id = calib_data['id']
                uncertainty = id_to_uncertainty[id]
                correctness = id_to_correctness[id]
                if uncertainty <= t_candidate:
                    m_cal += 1
                    if correctness == 0:
                        w_cal += 1
            if m_cal > w_cal:  # 有效样本才能用 Clopper-Pearson
                if args.upper_bound == 'CP':
                    r_upper = st.beta.ppf(1 - args.delta, w_cal + 1, m_cal - w_cal)
                elif args.upper_bound == 'HFD':
                    r_upper = w_cal / m_cal + math.sqrt(1 / (2 * m_cal) * math.log(1 / args.delta))
                elif args.upper_bound == 'EBB':
                    r_upper = w_cal / m_cal + math.sqrt(2 *  (w_cal / m_cal) * (1 -  w_cal / m_cal) * math.log(3 / args.delta) / m_cal) + 3 * math.log(3 / args.delta) / m_cal

                if r_upper <= alpha:
                    t_hat = t_candidate
                    low = mid + 1  # 尝试更大的 t
                else:
                    high = mid - 1
            else:
                high = mid - 1  # too few samples, move left
        if t_hat is None:
            print('The risk level is unmanageable!')
            continue
        else:
            print(f'Max threshold t_hat = {t_hat}')
            selected_num = 0
            error_num = 0
            for test_data in test_set:
                id = test_data['id']
                uncertainty = id_to_uncertainty[id]
                correctness = id_to_correctness[id]
                if uncertainty <= t_hat:
                    selected_num += 1
                    if correctness == 0:
                        error_num += 1
            risk = error_num / selected_num
            print(risk)
            multi_check_emr_list.append(risk)

    multi_check_emr_numpy = np.array(multi_check_emr_list)
    mean = np.mean(multi_check_emr_numpy)
    std = np.std(multi_check_emr_numpy)
    mean_list.append(mean)
    std_list.append(std)
    print('-'*20)
    print(mean)
    print(std)
    print('-' * 20)

print(alpha_list)
print(mean_list)
print(std_list)