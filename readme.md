# <div align="center">HyperGLLM: An Efficient Framework for Endpoint Threat Detection via Hypergraph-Enhanced Large Language Models<div>
<div align="center">
<p><strong>An efficient framework that introduces hypergraph reasoning into LLMs for malicious behavior detection in EDR logs.</strong></p>
<a href="https://huggingface.co/datasets/The_data_will_be_released_upon_acceptance_of_the_paper./" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Dataset-23999a.svg></a>
<a href="https://github.com/"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
<a><img alt="Static Badge" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

<h4 align="center">

<p>
<a href="#overview">Overview</a> |
<a href="#sparkles-features">Features</a> |  
<a href="#page_with_curl-changelog">Changelog</a> |
<a href="#rocket-quick-start">Quick-Start</a> |
<a href="#floppy_disk-dataset">Dataset</a> |
<a href="#raised_hands-faqs"> FAQs</a> 
</p>

## Overview

**HyperGLLM** is a novel detection framework that introduces hypergraph reasoning into LLMs. It first constructs an attribute-value level relation-aware graph to model low-order structural semantics while reducing textual redundancy. Then, it introduces a differential hypergraph module with multi-granularity clustering to capture high-order behavioral dependencies embedded in interleaved events and reinforce threat semantics. Finally, the hypergraph representations are aligned with an LLM for efficient contextual reasoning over potential malicious behaviors. We curate EDR3.6B-63F, a large-scale EDR dataset containing 3.6 billion events across 63 distinct behavior families. Extensive experiments demonstrate that HyperGLLM significantly outperforms state-of-the-art methods by reducing the false alarm rate to 1.67%, achieving 94.65% accuracy across 63 behavior families, and improving the modeling efficiency of LLMs on long EDR logs. Our framework and dataset provide a solid foundation for future research and support the development of advanced detection solutions in endpoint security.

<p align="center">
<img src="Imag/Method_Overview.png">
</p>

## :sparkles: Features

- **Framework**: We propose HyperGLLM, an efficient framework that introduces hypergraph reasoning into LLMs for malicious behavior detection in EDR logs, capturing both structural semantics and long-range temporal dependencies.
- **Structural Semantics & Temporal Dependencies**: We design an attribute-value level relation-aware graph and a differential hypergraph module with multi-granularity clustering to jointly model low- and high- order behavior semantics, thereby enhancing the semantic representation of threat behaviors.
- **EDR3.6B-63F Dataset**: We construct EDR3.6B-63F, a large-scale EDR dataset that serves as a high-quality benchmark for advancing AI-driven research in endpoint security, offering diverse behavior types and detailed event records.
- **Effective and Efficient**: Extensive experiments demonstrate that HyperGLLM consistently outperforms state-of-the-art baselines across multiple metrics while maintaining high inference efficiency.


## :page_with_curl: Changelog
[23/10/25] The project launched! 



## :rocket: Quick Start

We conduct training on eight NVIDIA H100 (80GB) GPUs and perform evaluation on a single GPU.

**Directory Overview**

- *requirements.txt* — Lists all dependencies required to reproduce this project.
- *datasets/* — Contains all datasets used in both the main experiments and the ablation study described in the paper.
- *experiments_main/* — Source code for reproducing the main experiments.
- *experiments_ablation/* — Source code for reproducing the ablation study experiments in the paper.
- *experiments_appendix/* — Code used for the appendix experiments included in the paper.

**Install Dependencies**

To reproduce our work, you need to have Python installed along with the required libraries. You can install the necessary dependencies using the following command:
```bash
pip install -r requirements.txt
```

**Reproducing the Main Experiment**

To reproduce the main experimental results:
1. cd to the main experiment directory:
```bash
cd experiments_main
```
2. Run the training and inference script:
```bash
sh run.sh
```
3. Obtain evaluation metrics:
```bash
python get_metrics.py
```
You can also obtain runtime efficiency metrics (GPU memory usage and Time-to-First-Token) by running:
```bash
python get_gpumu_tps.py
```
This provides performance metrics on an input of 1,024K tokens.

*To reproduce other experiments (e.g., the ablation study), cd to the corresponding directory (e.g., `cd experiments_ablation/Analysis_DHGNN`), run `sh run.sh` for training and inference, and obtain evaluation metrics with `python get_metrics.py`.


## :floppy_disk: Dataset

EDR3.6B-63F Dataset: [this repo](https://huggingface.co/datasets/The_data_will_be_released_upon_acceptance_of_the_paper./).

## :raised_hands: FAQs

## :bookmark: License

Our project is licensed under the [*MIT License*](./LICENSE).

## Citation
If you use the EDR3.6B-63F dataset in your research or find our method HyperGLLM inspiring, please consider citing our paper:

```bibtex
@inproceedings{
  title     = {HyperGLLM: An Efficient Framework for Endpoint Threat Detection via Hypergraph-Enhanced Large Language Models},
  year      = {2025},
}
```
