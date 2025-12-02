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
parser.add_argument('--dataset', default='commonsenseqa')
parser.add_argument('--max-length-of-generation', type=int, default=128)
parser.add_argument('--num-beams', type=int, default=5, help='for the most likely generation')
parser.add_argument('--num-generations-per-prompt', type=int, default=20, help='for sampling')
parser.add_argument('--top-p', type=float, default=0.9, help='for sampling')
parser.add_argument('--temperature', type=float, default=1.0, help='for sampling')
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
# load generation
with open(f'{run_name}/generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)
# load dataset
dataset = datasets.load_from_disk(f'{args.data_dir}/{args.dataset}_{model_name}')
id_to_prompt_mapping = dict(zip(dataset['id'], dataset['prompt']))  # dict{id: question, ...}
# ----------------------------------------------------------------------------------------------------------------------
# special tokens
eos_token_id = generative_tokenizer('. ')['input_ids'][1]
# eos_token_id (`Union[int, List[int]]`, *optional*):
#             The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
bad_words = ['### User:', '### Assistant:', '### System:', 'User:', 'Assistant:', 'System:', '###', '\n']
# bad_words_ids = [generative_tokenizer(bad_word, add_special_tokens=False)['input_ids'] for bad_word in bad_words]
bad_words_ids = [generative_tokenizer.encode(bad_word, add_special_tokens=False) for bad_word in bad_words]
# bad_words_ids = [[generative_tokenizer(bad_word)['input_ids'][1]] for bad_word in bad_words]
# bad_words_ids(`List[List[int]]`, *optional*):
#             List of token ids that are not allowed to be generated.
# ----------------------------------------------------------------------------------------------------------------------
# clean
results = []
for generation in tqdm.tqdm(generations):
    id = generation['id']
    prompt = id_to_prompt_mapping[id]
    most_likely_generated_text = generation['most_likely_generated_text']
    sampled_generated_texts = generation['sampled_generated_texts']
    options_letters = [temp[0] for temp in generation['options']]
    cleaned_generations = []
    # check most_likely_generated_text
    if most_likely_generated_text not in options_letters:
        print('Error: ', [most_likely_generated_text])
        max_attempts = 5
        # encode
        encode = generative_tokenizer(prompt)
        input_ids = torch.LongTensor(encode['input_ids']).cuda()
        attention_mask = torch.LongTensor(encode['attention_mask']).cuda()
        input_length = len(input_ids)
        input_ids = torch.reshape(input_ids, (-1, input_length))
        attention_mask = torch.reshape(attention_mask, (-1, input_length))
        while True:
            gen = generative_llm.generate(input_ids,
                                          attention_mask=attention_mask,
                                          num_beams=args.num_beams,
                                          do_sample=False,
                                          max_length=input_length + args.max_length_of_generation,
                                          eos_token_id=eos_token_id,
                                          bad_words_ids=bad_words_ids,
                                          temperature=0.1)
            gen_text = generative_tokenizer.decode(gen[0][input_length:], skip_special_tokens=True)
            if gen_text in options_letters:
                generation['most_likely_generated_text'] = gen_text
                break
            else:
                max_attempts -= 1
            if max_attempts == 0:
                break
    if generation['most_likely_generated_text'] not in options_letters:
        continue
    # check sampled_generated_texts
    error_generation = 0
    for sample_generation in sampled_generated_texts:
        if sample_generation in options_letters:
            cleaned_generations.append(sample_generation)
        else:
            print('Error: ', [sample_generation])
            error_generation += 1

    if error_generation >= (0.5*args.num_generations_per_prompt):
            continue
    # re-generate sampled_generated_texts
    if error_generation != 0:
        print(f"{error_generation} error generations.")
        # encode
        encode = generative_tokenizer(prompt)
        input_ids = torch.LongTensor(encode['input_ids']).cuda()
        attention_mask = torch.LongTensor(encode['attention_mask']).cuda()
        input_length = len(input_ids)
        input_ids = torch.reshape(input_ids, (-1, input_length))
        attention_mask = torch.reshape(attention_mask, (-1, input_length))

        while True:
            gen = generative_llm.generate(input_ids,
                                          attention_mask=attention_mask,
                                          do_sample=True,
                                          num_return_sequences=1,  # <= num_beams
                                          num_beams=1,  # greedy search
                                          max_length=input_length + args.max_length_of_generation,
                                          eos_token_id=eos_token_id,
                                          bad_words_ids=bad_words_ids,
                                          temperature=args.temperature,
                                          top_p=args.top_p)
            gen_text = generative_tokenizer.decode(gen[0][input_length:], skip_special_tokens=True)
            if gen_text in options_letters:
                cleaned_generations.append(gen_text)
                error_generation -= 1
                print(f"{error_generation} error generations.")

            if error_generation == 0:
                break
    # re-check
    assert generation['most_likely_generated_text'] in options_letters
    generation['sampled_generated_texts'] = cleaned_generations
    assert len(generation['sampled_generated_texts']) == args.num_generations_per_prompt
    for cleaned_generation in generation['sampled_generated_texts']:
        assert cleaned_generation in options_letters

    results.append(generation)

# save
with open(f'{run_name}/cleaned_generations.pkl', 'wb') as record_file:
    pickle.dump(results, record_file)
print('Record saved to ', f'{run_name}/cleaned_generations.pkl')
