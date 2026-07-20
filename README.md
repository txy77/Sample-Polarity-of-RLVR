# Rethinking Sample Polarity in Reinforcement Learning with Verifiable Rewards

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ACL%202026-red)](https://aclanthology.org/2026.acl-long.134/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

Official implementation of the ACL 2026 paper **"Rethinking Sample Polarity in Reinforcement Learning with Verifiable Rewards"**.

> Xinyu Tang\*, Yuliang Zhan\*, Zhixun Li\*, Wayne Xin Zhao†, Zhenduo Zhang, Zujie Wen, Zhiqiang Zhang, Jun Zhou
>
> Gaoling School of Artificial Intelligence, Renmin University of China · The Chinese University of Hong Kong · Ant Group

## 📖 Overview

In RLVR (Reinforcement Learning with Verifiable Rewards), policies are updated with both **positive** and **negative** self-generated rollouts — two distinct *sample polarities*. This work provides a systematic study of how each polarity shapes RLVR training, and proposes a new advantage-shaping method based on the findings.

**Key findings:**

- **Positive samples sharpen, negative samples discover.** Positive-only reinforcement (PSR) amplifies existing correct reasoning patterns, reduces entropy, and shortens responses — eventually collapsing into reward hacking. Negative-only reinforcement (NSR) encourages exploration of new reasoning paths, maintains entropy, and lengthens responses — but can drift into low-probability regions and produce garbled output.
- **Both polarities are essential.** Training with a single polarity harms reasoning ability and the reasoning boundary across different base LLMs; negative samples are especially important for preserving generalization.
- **Polarity-level shaping:** up-weighting positive advantages speeds up reward growth but narrows exploration; up-weighting negative advantages broadens exploration but slows reward improvement. Training dynamics are governed by the *relative ratio* of positive-to-negative advantage, not the absolute values.
- **Token-level shaping:** weighting tokens by entropy or probability has *opposite* effects in positive vs. negative samples. In particular, **low-probability tokens in positive samples** and **high-probability tokens in negative samples** are key to maintaining entropy and exploration in early training.

**Method — A3PO.** Building on these insights, we propose an **A**daptive and **A**symmetric token-level **A**dvantage shaping method for **P**olicy **O**ptimization (A3PO). It assigns larger advantages to low-probability positive tokens and high-probability negative tokens early in training to encourage exploration, then adaptively decays these weights as training progresses to avoid training–inference mismatch and ensure stable convergence.

## 📊 Main Results

Average accuracy on five reasoning benchmarks (AIME24, AIME25, MATH500, GPQA, LiveCodeBench):

| Method | Qwen2.5-7B-Math | Qwen3-8B-Base | DeepSeek-R1-Distill-Qwen-7B |
| --- | :---: | :---: | :---: |
| GRPO | 35.0 | 42.5 | 58.9 |
| DAPO | 36.3 | 44.1 | 60.1 |
| DAPO w/ Fork Tokens | 37.7 | 45.1 | 60.6 |
| W-REINFORCE | 37.4 | 45.3 | 60.9 |
| Lp-Reg | 37.8 | 45.6 | 61.2 |
| **A3PO (ours)** | **40.4** | **48.7** | **63.4** |

Full per-benchmark numbers, standard deviations, and significance tests are reported in Table 2 of the paper.

## 🗂️ Repository Structure

This repository is built on [verl](https://github.com/volcengine/verl) (v0.4.0.dev). The paper-specific modifications are:

```
├── debug/
│   ├── run_qwen3-4b-psr.sh       # Positive Sample Reinforcement (PSR)
│   ├── run_qwen3-4b-nsr.sh       # Negative Sample Reinforcement (NSR)
│   └── run_qwen3-4b_grpo.sh      # Standard GRPO baseline
├── verl/trainer/ppo/
│   ├── core_algos.py             # `psr_nsr` advantage estimator (PSR / NSR / polarity-weighted)
│   └── ray_trainer.py            # Polarity-level & token-level advantage shaping
└── verl/trainer/config/
    └── ppo_trainer.yaml          # New `algorithm.*` options (see below)
```

## 🚀 Getting Started

### Installation

The environment follows verl. See the [verl installation guide](https://verl.readthedocs.io/en/latest/start/install.html) for details.

```bash
conda create -n rlvr python=3.10 -y
conda activate rlvr

git clone https://github.com/txy77/Sample-Polarity-of-RLVR.git
cd Sample-Polarity-of-RLVR
pip install -e .
pip install vllm ray
```

### Data Preparation

- **Training:** [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) (parquet format).
- **Validation:** AIME 2024 / AIME 2025 (parquet format).

Preprocessing scripts for common math datasets are available under `examples/data_preprocess/`.

### Training

Edit the environment variables at the top of the scripts in `debug/` (model path, data paths, save paths), then launch:

```bash
# Positive Sample Reinforcement (PSR)
bash debug/run_qwen3-4b-psr.sh

# Negative Sample Reinforcement (NSR)
bash debug/run_qwen3-4b-nsr.sh

# Standard GRPO
bash debug/run_qwen3-4b_grpo.sh
```

### Key Configuration Options

The sample-polarity experiments are controlled by the following `algorithm.*` options (see `verl/trainer/config/ppo_trainer.yaml`):

| Option | Description |
| --- | --- |
| `adv_estimator` | Set to `psr_nsr` for single-polarity / polarity-weighted training, or `grpo` for GRPO with token-level shaping |
| `advantage` | Sample polarity: `positive` (PSR), `negative` (NSR), or `weighted` (polarity-level advantage shaping) |
| `positive_advantage_weight` / `negative_advantage_weight` | Polarity-level advantage scaling factors (e.g., `P2N1` in the paper = positive ×2, negative ×1) |
| `token_weighted_metric` | Token-level shaping metric: `prob` or `entropy` |
| `token_weighted_{positive,negative}_{high,low}_num_ratio` | Fraction of tokens (by metric ranking) to reweight for each polarity |
| `token_weighted_{positive,negative}_{high,low}_scale` | Advantage scaling factor applied to the selected tokens |

For example, amplifying the advantage of the top-20% highest-probability tokens in negative samples by ×5 corresponds to `token_weighted_metric=prob`, `token_weighted_negative_high_num_ratio=0.2`, `token_weighted_negative_high_scale=5`.

## 📝 Citation

If you find this work helpful, please cite:

```bibtex
@inproceedings{tang-etal-2026-rethinking,
    title = "Rethinking Sample Polarity in Reinforcement Learning with Verifiable Rewards",
    author = "Tang, Xinyu and Zhan, Yuliang and Li, Zhixun and Zhao, Xin and
              Zhang, Zhenduo and Wen, Zujie and Zhang, Zhiqiang and Zhou, Jun",
    booktitle = "Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.134/",
    doi = "10.18653/v1/2026.acl-long.134",
    pages = "2928--2954"
}
```

## 🙏 Acknowledgements

This codebase is built on [verl](https://github.com/volcengine/verl), an open-source RLHF/RLVR training framework from the verl community. We thank the authors for their excellent work.
