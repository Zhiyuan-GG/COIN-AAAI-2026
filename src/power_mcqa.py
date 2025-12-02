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
parser.add_argument('--delta', type=float, default=0.05, help='significance level')
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
# load logit
with open(f'{run_name_logit}/logit.pkl', 'rb') as record_file:
    id_to_logit = pickle.load(record_file)
# ----------------------------------------------------------------------------------------------------------------------
generations = generations[:args.apply]
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
def compute_bh_selected_error_rate(p_value_list, correctness_list, alpha):
    m = len(p_value_list)
    p_value_array = np.array(p_value_list)
    correctness_array = np.array(correctness_list, dtype=bool)

    # Benjamini-Hochberg procedure
    sorted_indices = np.argsort(p_value_array)  # p-value从小到大排序之后的 indices
    sorted_p_values = p_value_array[sorted_indices]
    bh_thresholds = alpha * (np.arange(1, m + 1) / m)
    selected_flags = sorted_p_values <= bh_thresholds

    if selected_flags.any():
        k_star = np.max(np.where(selected_flags)[0])
        selected_indices = sorted_indices[:k_star + 1]
    else:
        selected_indices = np.array([], dtype=int)

    # FDR calculation (empirical error rate)
    if len(selected_indices) > 0:
        num_incorrect_in_selected = (~correctness_array[selected_indices]).sum()
        fdr = num_incorrect_in_selected / len(selected_indices)
    else:
        fdr = None

    # Power calculation
    num_true_aligned = correctness_array.sum()
    if num_true_aligned > 0:
        num_selected_true_aligned = correctness_array[selected_indices].sum()
        power = num_selected_true_aligned / num_true_aligned
    else:
        power = None

    return fdr, power

qa_num = len(generations)
num_calib = int(qa_num * args.split_ratio)
num_test = qa_num - num_calib
# risk level list
alpha_list = [0.1, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
cp_mean_list = []
cp_std_list = []
hfd_mean_list = []
hfd_std_list = []
ca_mean_list = []
ca_std_list = []
for alpha in alpha_list:
    cp_multi_check_power_list = []
    hfd_multi_check_power_list = []
    ca_multi_check_power_list = []
    for epoch in tqdm.tqdm(range(args.multi_check)):
        # for reproduction
        random.seed(epoch)
        calib_set = random.sample(generations, num_calib)
        test_set = [gen for gen in generations if gen not in calib_set]
        # --------------------------------------------------------------------------------------------------------------
        # for threshold selection: t-hat
        min_uncertainty = min([id_to_uncertainty[d['id']] for d in calib_set])
        max_uncertainty = max([id_to_uncertainty[d['id']] for d in calib_set])
        sorted_uncertainties = np.linspace(min_uncertainty, max_uncertainty, num_calib).tolist()
        # for cp
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
            if m_cal > w_cal:
                r_upper = st.beta.ppf(1 - args.delta, w_cal + 1, m_cal - w_cal)
                if r_upper <= alpha:
                    t_hat = t_candidate
                    low = mid + 1  # 尝试更大的 t
                else:
                    high = mid - 1
            else:
                high = mid - 1  # too few samples, move left
        if t_hat is None:
            print('The risk level is unmanageable!')
            cp_power = 0
        else:
            selected_num = 0
            error_num = 0
            cp_correctness_list = []
            cp_selected_correctness_list = []
            for test_data in test_set:
                id = test_data['id']
                uncertainty = id_to_uncertainty[id]
                correctness = id_to_correctness[id]
                cp_correctness_list.append(correctness)
                if uncertainty <= t_hat:
                    cp_selected_correctness_list.append(correctness)
                    selected_num += 1
                    if correctness == 0:
                        error_num += 1
            risk = error_num / selected_num
            if risk <= alpha:
                cp_power = sum(cp_selected_correctness_list) / max(sum(cp_correctness_list), 1)
            else:
                cp_power = 0
                print('The risk level is unmanageable!')
        # for hfd
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
            if m_cal > w_cal:
                r_upper = w_cal / m_cal + math.sqrt(1 / (2 * m_cal) * math.log(1 / args.delta))
                if r_upper <= alpha:
                    t_hat = t_candidate
                    low = mid + 1  # 尝试更大的 t
                else:
                    high = mid - 1
            else:
                high = mid - 1  # too few samples, move left
        if t_hat is None:
            print('The risk level is unmanageable!')
            hfd_power = 0
        else:
            selected_num = 0
            error_num = 0
            hfd_correctness_list = []
            hfd_selected_correctness_list = []
            for test_data in test_set:
                id = test_data['id']
                uncertainty = id_to_uncertainty[id]
                correctness = id_to_correctness[id]
                hfd_correctness_list.append(correctness)
                if uncertainty <= t_hat:
                    hfd_selected_correctness_list.append(correctness)
                    selected_num += 1
                    if correctness == 0:
                        error_num += 1
            risk = error_num / selected_num
            if risk <= alpha:
                hfd_power = sum(hfd_selected_correctness_list) / max(sum(hfd_correctness_list), 1)
            else:
                hfd_power = 0
                print('The risk level is unmanageable!')
        # for ca
        p_value_list = []
        correctness_list = []
        for test_data in test_set:
            test_id = test_data['id']
            test_uncertainty = id_to_uncertainty[test_id]
            test_correctness = id_to_correctness[test_id]
            correctness_list.append(test_correctness)
            count = 0
            for calib_data in calib_set:
                calib_id = calib_data['id']
                calib_uncertainty = id_to_uncertainty[calib_id]
                calib_correctness = id_to_correctness[calib_id]
                if calib_uncertainty <= test_uncertainty and calib_correctness == 0:
                    count += 1
            p_value = (count + 1) / (len(calib_set) + 1)
            p_value_list.append(p_value)

        risk, power = compute_bh_selected_error_rate(p_value_list, correctness_list, alpha)
        if risk is not None and risk <= alpha and power is not None:
            ca_power = power
        else:
            ca_power = 0

        cp_multi_check_power_list.append(cp_power)
        hfd_multi_check_power_list.append(hfd_power)
        ca_multi_check_power_list.append(ca_power)

    cp_multi_check_power_numpy = np.array(cp_multi_check_power_list)
    cp_mean = np.mean(cp_multi_check_power_numpy)
    cp_std = np.std(cp_multi_check_power_numpy)
    cp_mean_list.append(cp_mean)
    cp_std_list.append(cp_std)

    hfd_multi_check_power_numpy = np.array(hfd_multi_check_power_list)
    hfd_mean = np.mean(hfd_multi_check_power_numpy)
    hfd_std = np.std(hfd_multi_check_power_numpy)
    hfd_mean_list.append(hfd_mean)
    hfd_std_list.append(hfd_std)

    ca_multi_check_power_numpy = np.array(ca_multi_check_power_list)
    ca_mean = np.mean(ca_multi_check_power_numpy)
    ca_std = np.std(ca_multi_check_power_numpy)
    ca_mean_list.append(ca_mean)
    ca_std_list.append(ca_std)

print(alpha_list)
print(cp_mean_list)
print(cp_std_list)
print(hfd_mean_list)
print(hfd_std_list)
print(ca_mean_list)
print(ca_std_list)