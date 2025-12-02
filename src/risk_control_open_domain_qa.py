import argparse
import collections
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
                    default=r'Llama-3.1-8B-Instruct',
                    help='model version')
parser.add_argument('--dataset', default='triviaqa')
parser.add_argument('--max-length-of-generation', type=int, default=128)
parser.add_argument('--num-beams', type=int, default=5, help='for the most likely generation')
parser.add_argument('--num-generations-per-prompt', type=int, default=10, help='for sampling')
parser.add_argument('--top-p', type=float, default=0.9, help='for sampling')
parser.add_argument('--temperature', type=float, default=1.0, help='for sampling')
# ----------------------------------------------------------------------------------------------------------------------
# for risk control
parser.add_argument('--correctness-method', type=str, default='entailment', help='similarity, entailment')
parser.add_argument('--correctness-threshold', type=float, default=0.7, help='for correctness evaluation')
parser.add_argument('--uncertainty', type=str, default='seb', help='[lnpe, pe, seb, sew, deg, ecc, eigv]')
parser.add_argument('--samples', type=int, default=10, help='num generations for UQ')
parser.add_argument('--split-ratio', type=float, default=0.5, help='for calib and test num')
parser.add_argument('--multi-check', type=int, default=100, help='for multiple split and check')
parser.add_argument('--alpha', type=list, default=[0.1, 0.2, 0.3, 0.4, 0.5], help='risk level')
parser.add_argument('--delta', type=float, default=0.05, help='significance level')
parser.add_argument('--upper-bound', type=str, default='CP', help='CP (Clopper–Pearson) OR HFD (Hoeffding) OR EBB')
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
# load most_likely_generated_text and sampled_generated_texts
with open(f'{run_name}/cleaned_generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)
# load logit
with open(f'{run_name}/entropies.pkl', 'rb') as record_file:
    id_to_entropies = pickle.load(record_file)
# load similarity scores for correctness
with open(f'{run_name}/most_likely_generation_similarity.pkl', 'rb') as record_file:
    id_to_most_likely_generation_similarity = pickle.load(record_file)
# load llm judge for correctness
with open(f'{run_name}/most_likely_generation_entailment_score.pkl', 'rb') as record_file:
    id_to_most_likely_generation_entailment_score = pickle.load(record_file)
# load clusters for semantic uncertainty
with open(f'{run_name}/sampled_generations_cluster.pkl', 'rb') as record_file:
    id_to_sampled_generations_cluster = pickle.load(record_file)
# load sampled_generations_similarity
with open(f'{run_name}/sampled_generations_similarity.pkl', 'rb') as record_file:
    id_to_sampled_generations_similarity = pickle.load(record_file)
# ----------------------------------------------------------------------------------------------------------------------
temp = []
for gen in tqdm.tqdm(generations):
    id = gen['id']
    if args.correctness_method == 'entailment':
        if id in id_to_most_likely_generation_entailment_score and id in id_to_entropies:
            temp.append(gen)
    elif args.correctness_method == 'similarity':
        if id in id_to_most_likely_generation_similarity and id in id_to_entropies:
            temp.append(gen)
generations = temp
# ----------------------------------------------------------------------------------------------------------------------
def compute_W(similarity_list):
    """
    构造加权对称邻接矩阵 W
    """
    similarity_array = np.array(similarity_list)
    W = (similarity_array + similarity_array.T) / 2
    return W


def compute_degree_matrix(W):
    """
    构造度矩阵 D
    """
    degrees = np.sum(W, axis=1)
    D = np.diag(degrees)
    return D


def compute_L(W, D):
    """
    构造归一化图拉普拉斯矩阵 L
    """
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    L = np.identity(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    return L


def eigv_uncertainty(similarity_list):
    W = compute_W(similarity_list)
    D = compute_degree_matrix(W)
    L = compute_L(W, D)
    eigenvalues = np.linalg.eigvalsh(L)
    U_EigV = np.sum(np.maximum(0, 1 - eigenvalues))
    return U_EigV


def deg_uncertainty(similarity_list):
    W = compute_W(similarity_list)
    D = compute_degree_matrix(W)
    m = W.shape[0]
    U_Deg = (m - np.trace(D)) / (m ** 2)
    return U_Deg


def ecc_uncertainty(similarity_list, k=None):
    W = compute_W(similarity_list)
    D = compute_degree_matrix(W)
    L = compute_L(W, D)

    eigenvalues, eigenvectors = np.linalg.eigh(L)
    m = W.shape[0]
    if k is None:
        k = min(5, m)  # 默认取前5个特征向量，防止太高维

    V = eigenvectors[:, :k]
    v_mean = np.mean(V, axis=0)
    V_centered = V - v_mean
    U_Ecc = np.linalg.norm(V_centered)
    return U_Ecc

# uncertainty and correctness
id_to_uncertainty = {}
id_to_correctness = {}
for gen in tqdm.tqdm(generations):
    id = gen['id']
    most_likely_generation_similarity = id_to_most_likely_generation_similarity[id]
    most_likely_generation_entailment_score = id_to_most_likely_generation_entailment_score[id]
    sampled_generations_cluster = id_to_sampled_generations_cluster[id]
    entropies = id_to_entropies[id]
    similarity_list = id_to_sampled_generations_similarity[id]
    # correctness
    if args.correctness_method == 'entailment':
        correctness = int(most_likely_generation_entailment_score >= args.correctness_threshold)
        id_to_correctness[id] = correctness
    elif args.correctness_method == 'similarity':
        correctness = int(most_likely_generation_similarity >= args.correctness_threshold)
        id_to_correctness[id] = correctness
    # uncertainty
    if args.uncertainty == 'lnpe':
        lnpe_list = []
        for ent in entropies:
            lnpe = torch.tensor(ent).mean()
            lnpe_list.append(lnpe)
        uncertainty = torch.tensor(lnpe_list[:args.samples]).mean().item()
    elif args.uncertainty == 'pe':
        pe_list = []
        for ent in entropies:
            pe = torch.tensor(ent).sum()
            pe_list.append(pe)
        uncertainty = torch.tensor(pe_list[:args.samples]).mean().item()
    elif args.uncertainty == 'seb':
        ans_cluster_idx = sampled_generations_cluster[:args.samples]
        cluster_dis = collections.Counter(ans_cluster_idx)
        uncertainty = -sum(
            (cnt / len(ans_cluster_idx)) * math.log2(cnt / len(ans_cluster_idx)) for cnt in
            cluster_dis.values())
    elif args.uncertainty == 'ecc':
        uncertainty = ecc_uncertainty(similarity_list)
    elif args.uncertainty == 'eigv':
        uncertainty = eigv_uncertainty(similarity_list)
    elif args.uncertainty == 'deg':
        uncertainty = deg_uncertainty(similarity_list)
    # elif args.uncertainty == 'sew':
    #     llh_shift = torch.tensor(5.0)
    #     ans_cluster_idx = torch.tensor(sampled_generations_cluster[:args.samples])
    #     lnpe_list = []
    #     for ent in entropies[:args.samples]:
    #         lnpe = torch.tensor(ent).mean()
    #         lnpe_list.append(lnpe)
    #     gen_entropy = torch.tensor(lnpe_list)
    #     semantic_cluster_entropy = []
    #     for semantic_id in torch.unique(ans_cluster_idx):
    #         semantic_cluster_entropy.append(torch.logsumexp(-1 * gen_entropy[ans_cluster_idx == semantic_id], dim=0))
    #     semantic_cluster_entropy = torch.tensor(semantic_cluster_entropy) - llh_shift
    #     semantic_cluster_entropy = - torch.sum(semantic_cluster_entropy, dim=0) / torch.tensor(
    #         semantic_cluster_entropy.shape[0])
    #     uncertainty = torch.mean(semantic_cluster_entropy).item()

    id_to_uncertainty[id] = uncertainty
# ----------------------------------------------------------------------------------------------------------------------
qa_num = len(generations)
num_calib = int(qa_num * args.split_ratio)
num_test = qa_num - num_calib
# risk level list
alpha_list = args.alpha
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
                    else:
                        print(alpha)
                        print(uncertainty)
                        print(t_hat)
                        print(test_data['question'])
                        print(test_data['answer'])
                        print(test_data['most_likely_generated_text'])
                        exit()
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