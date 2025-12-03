# [AAAI 2026] COIN: Uncertainty-Guarding Selective Question Answering for Foundation Models with Provable Risk Guarantees
- **Authors:** Zhiyuan Wang, Jinhao Duan, Qingni Wang, Xiaofeng Zhu, Tianlong Chen, Xiaoshuang Shi*, Kaidi Xu*
- [**arXiv**](https://arxiv.org/abs/2506.20178)

![Overview of COIN](overview.png)
- **Step 1:** *Given an initialized threshold t, we perform selective prediction on the calibration set and estimate the empirical false discovery rate (FDR).*
- **Step 2:** *Establish the (1-&delta;) upper confidence bound of the system risk.*
- **Step 3:** *Adjust the threshold t such that the bound does not exceed &alpha;, and select the largest t to filter out samples at test time.*

![Inplementation](Implementation.png)


### Parse Data
```shell
python parse_commonsense_qa.py --cache-model LocalPath --generate-model ModelVersion --few-shot-num 1 --max-num 10000 
python parse_triviaqa.py --cache-model LocalPath --generate-model ModelVersion --few-shot-num 1 --max-num 2000 
```
- LocalPath: Local path for saving downloaded models
- ModelVersion: Model version, such as DeepSeek-R1-Distill-Qwen-7B

### Generate and Clean
```shell
python generate.py --cache-model LocalPath --generate-model ModelVersion --dataset commonsenseqa --num-generations-per-prompt 20
python generate.py --cache-model LocalPath --generate-model ModelVersion --dataset triviaqa --num-generations-per-prompt 10
python clean_mcqa.py --cache-model LocalPath --generate-model ModelVersion --dataset commonsenseqa --num-generations-per-prompt 20
python clean_open_domain_qa.py --cache-model LocalPath --generate-model ModelVersion --dataset triviaqa --num-generations-per-prompt 10 
```

### Prepare for Correctness and Uncertainty
```shell
python get_mcqa_logit.py --cache-model LocalPath --generate-model ModelVersion --dataset commonsenseqa
python get_open_domain_qa_logits.py --cache-model LocalPath --generate-model ModelVersion --dataset triviaqa --num-generations-per-prompt 10
python open_domain_qa_clustering.py --cache-model LocalPath --generate-model ModelVersion --dataset triviaqa --num-generations-per-prompt 10 --infer-model NLIModel
python open_domain_qa_mst_entailment.py --cache-model LocalPath --generate-model ModelVersion --dataset triviaqa --num-generations-per-prompt 10 --infer-model NLIModel
python open_domain_qa_mst_similarity.py  --cache-model LocalPath --generate-model ModelVersion --dataset triviaqa --num-generations-per-prompt 10 --similarity-model SentenceSimilarityModel
python open_domain_qa_similarity.py --cache-model LocalPath --generate-model ModelVersion --dataset triviaqa --num-generations-per-prompt 10 --similarity-model SentenceSimilarityModel
```
- NLIModel: Natural language inference model, such as deberta-large-mnli (contradiction neutrality entailment)
- SentenceSimilarityModel: stsb-distilroberta-base or stsb-roberta-large

### Risk Control and Power
```shell
python risk_control_mcqa.py --cache-model LocalPath --generate-model ModelVersion --dataset commonsenseqa --num-generations-per-prompt 20 --apply 6000 --uncertainty white --split-ratio 0.5 --multi-check 100 --alpha [0.1, 0.2, 0.3] --delta 0.05 --upper-bound CP
python risk_control_mcqa.py --cache-model LocalPath --generate-model ModelVersion --dataset commonsenseqa --num-generations-per-prompt 20 --apply 6000 --uncertainty white --split-ratio 0.5 --multi-check 100 --alpha [0.1, 0.2, 0.3] --delta 0.05 --upper-bound HFD
python risk_control_open_domain_qa.py --cache-model LocalPath --generate-model ModelVersion --dataset triviaqa --num-generations-per-prompt 10 --correctness-method similarity --correctness-threshold 0.7 --uncertainty seb --samples 10 --split-ratio 0.5 --alpha [0.1, 0.2, 0.3] --delta 0.05 --upper-bound CP
python risk_control_open_domain_qa.py --cache-model LocalPath --generate-model ModelVersion --dataset triviaqa --num-generations-per-prompt 10 --correctness-method similarity --correctness-threshold 0.7 --uncertainty seb --samples 10 --split-ratio 0.5 --alpha [0.1, 0.2, 0.3] --delta 0.05 --upper-bound HFD
python power_mcqa.py --cache-model LocalPath --generate-model ModelVersion --dataset commonsenseqa --num-generations-per-prompt 20 --apply 6000 --uncertainty white --split-ratio 0.5 --multi-check 100 --alpha [0.1, 0.2, 0.3] --delta 0.05
python power_open_domain_qa.py --cache-model LocalPath --generate-model ModelVersion --dataset triviaqa --num-generations-per-prompt 10 --correctness-method similarity --correctness-threshold 0.7 --uncertainty seb --samples 10 --split-ratio 0.5 --alpha [0.1, 0.2, 0.3] --delta 0.05
```