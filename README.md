# Argument Quality Assessment with Large Language Models: A Pairwise Bradley-Terry Approach

This repository contains the source code for all experiments carried out in the
paper entitled "Argument Quality Assessment with Large Language Models: A
Pairwise Bradley-Terry Approach". We also release the generations produced by
each evaluated LLM, prompt, trial, and dimension.

## TLDR

In this repository, we release:

- The generations for all pairwise judgments produced by 12 LLMs, 3 prompting
  strategies, 3 trial runs, 3 dimensions (totalling 324 runs), along with their
  aggregated annotations in `data/webis_only_args_all_pred_llms.csv` and
  `data/webis_only_args_bt_scores.csv`. The first file contains the judgments
  for each pairwise comparison configuration, while the second contains the
  resulting Bradley-Terry rankings obtained from those comparisons.
- Reproducibility scripts for each table, plot, and result in the paper (in
  `analysis/`).
- A generation script for the pairwise judgment labels in
  `experiments/predict_vect.py`.
- The prompts, LLMs, and dimensions evaluated in `experiments/conf/prompt`,
  `experiments/conf/llm`, and `experiments/conf/dim`.

## Installation

Create a python environment on your machine. You can use the environment manager
of preference. For our experiments we used `venv`. Then proceed to install the
requirements of the project.

```bash
python -m venv arg_quality_with_llms_venv
source arg_quality_with_llms_venv/bin/activate
pip install -r requirements.txt
```

You must create as well a `.env` file that contains your READ token to
huggingface that allows you access to the following models:

- Mistral-7B: `mistralai/Mistral-7B-Instruct-v0.3`
- Llama-8B: `meta-llama/Llama-3.1-8B-Instruct`
- Olmo2-7B: `allenai/OLMo-2-1124-7B-Instruct`
- Qwen2.5-7B: `Qwen/Qwen2.5-7B-Instruct`
- Command-r-7B: `CohereLabs/c4ai-command-r7b-12-2024`
- Mixtral-8x7B: `mistralai/Mixtral-8x7B-Instruct-v0.1`
- Mistral-22B: `mistralai/Mistral-Small-Instruct-2409`
- Olmo2-32B: `allenai/OLMo-2-0325-32B-Instruct`
- Mixtral-8x22B: `mistralai/Mixtral-8x22B-Instruct-v0.1`
- Llama-70B: `meta-llama/Llama-3.3-70B-Instruct`
- Qwen2.5-72B: `Qwen/Qwen2.5-72B-Instruct`
- Command-r-104B: `CohereLabs/c4ai-command-r-plus-08-2024`

In order to gain access to these models you can request it through huggingface
and fill in their corresponding access forms. In general, they have a quick
response time.

The `.env` file should contain the following environment variable:

```
HF_TOKEN=<your-token>
```

## Reproducibility of Experiments

The `experiments/` directory contains all the generation scripts that perform
the 324 runs, while the `analysis/` directory contains all the scripts necessary
to reproduce the tables, plots, and results presented in the paper.

`experiments/` is mainly composed of:

- `predict_vect.py`, the main generation script.
- `run_job.sh`, a script that encapsulates `predict_vect.py` to control
  parameters, configuration files, run names, among others.
- `run_all.sh`, a script that uses `run_job.sh` to execute the 324 runs of our
  experiments.
- `utils.py` and `gen_pairs.py`, helper scripts used to preprocess the data and
  generate pairwise comparisons from the original datasets released in the
  Webis-ArgQuality-20 corpus.
- `conf/`, which contains the configuration files used for the evaluated LLMs,
  prompts, and dimensions.

`analysis/` is mainly composed of:

- `00_bt_ranking.py`, which takes the generations produced by `predict_vect.py`
  and saved in `data/webis_only_args_all_pred_llms.csv`, and computes the
  Bradley-Terry (BT) rankings already provided in
  `data/webis_only_args_bt_scores.csv` for all configurations.
- `01_llms_vs_experts_performance.py`, which uses
  `data/webis_only_args_all_pred_llms.csv` and
  `data/webis_only_args_bt_scores.csv` to compute the results reported in Tables
  1 and 2 of the paper.
- `02_best_llms_vs_other_llms.py`, which uses
  `data/webis_only_args_all_pred_llms.csv` and
  `data/webis_only_args_bt_scores.csv` to compute the results reported in Table
  3 of the paper.
- `03_prediction_variability.py`, which uses
  `data/webis_only_args_all_pred_llms.csv` to compute the prediction variability
  reported in Table 4 of the paper.
- `04_bt_distplots.py`, which uses `data/webis_only_args_bt_scores.csv` to
  generate the BT score distributions reported in Figure 1 for the best model
  configuration.
- `05_few_shot_examples.py`, which contains the code used to select the few-shot
  examples employed in our experiments.

## Extra tables for footnotes 4 and 5 of the paper

Given the 12 LLMs, 3 prompts, and 3 dimensions (108 total rows), we did not
include the full table in either the main paper or the appendix. Instead, we
report the average scores across the three dimensions (36 rows) and provide the
complete 108-row table as supplementary material in this repository.

These results are saved in `supplementary_results/` namely
`llm_performance_per_llm_prompt_dim.csv` for footnote 4, and
`best_llm_vs_others_per_llm_prompt_dim.csv` for footnote 5.