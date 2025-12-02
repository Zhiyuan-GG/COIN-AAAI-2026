# [AAAI 2026] COIN: Uncertainty-Guarding Selective Question Answering for Foundation Models with Provable Risk Guarantees
- **Authors:** Zhiyuan Wang, Jinhao Duan, Qingni Wang, Xiaofeng Zhu, Tianlong Chen, Xiaoshuang Shi*, Kaidi Xu*
- [**arXiv**](https://arxiv.org/abs/2506.20178)

![Overview of COIN](overview.png)
- **Step one:** *Given an initialized threshold t, we perform selective prediction on the calibration set and estimate the empirical false discovery rate (FDR).*
- **Step two:** *Establish the (1-&delta;) upper confidence bound of the system risk.*
- **Step three:** *Adjust the threshold t such that the bound does not exceed &alpha;, and select the largest t to filter out samples at test time.*

![Inplementation](Implementation.png)





