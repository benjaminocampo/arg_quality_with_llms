# %%
import pandas as pd
import numpy as np
from scipy.stats import zscore, pearsonr, spearmanr, kendalltau
from pathlib import Path

df_scores = pd.read_csv("../data/webis_only_args_bt_scores.csv")
# %% 
df_scores.columns
# %%
MODEL_NAMES = [
    "mistral-7B-small",
    "llama-8B-small",
    #"olmo2-7B-small",
    "qwen2.5-7B-small",
    "commandr-7B-small",
    "mixtral-8x7B-medium",
    "mistral-22B-medium",
    "olmo2-32B-medium",
    "mixtral-8x22B-medium",
    "llama-70B-big",
    "qwen2.5-72B-big",
    "commandr-104B-big"
]
DIMENSIONS   = ["logic", "rhetoric", "dialectic"]
PROMPTS = ["zero_shot"]
# %%
uncovered_args = []
for model_name in MODEL_NAMES:
    for dim in DIMENSIONS:
        for prompt in PROMPTS:
            run_name = f"{model_name}_{dim}_{prompt}_bt_zscore"
            uncovered_args.append(df_scores.loc[df_scores[run_name].isna(), "id"])
# %%
# There are some uncovered args that a quality score could not be computed
# because of the selection in the comparisons. Only 2 of them were uncovered. It
# happened in all LLMs. The comparison was done by the original Webis dataset.
for i in range(len(uncovered_args) - 1):
    assert all(uncovered_args[i] == uncovered_args[i+1])
# %%
len(uncovered_args[0])
# %%
uncovered_args[0]
# %%
# Keep those were an score could be calculated
df_scores = df_scores[~df_scores["id"].isin(uncovered_args[0].tolist())].copy()
# %%
df_scores = df_scores.reset_index(drop=True)
# %%
results = {}
for model_name in MODEL_NAMES:
    for dim in DIMENSIONS:
        for prompt in PROMPTS:
            run_name = f"{model_name}_{dim}_{prompt}_bt_zscore"
            
            # Adding suffix: RhetoricAL, LogicAL, DialecticAL
            expert_z = zscore(df_scores[f"{dim.capitalize()}al Quality"])
            llm_z = zscore(df_scores[run_name])
            expert_score = df_scores[f"{dim.capitalize()}al Quality"]
            llm_score = df_scores[run_name]

            pearson  = pearsonr(expert_z, llm_z).statistic
            spearman = spearmanr(expert_score, llm_score).correlation 
            kendall  = kendalltau(expert_score, llm_score).correlation  
            mae      = np.mean(np.abs(expert_z - llm_z))
            rmse     = np.sqrt(np.mean((expert_z - llm_z)**2))

            results[f"{model_name}_{dim}_{prompt}"] = {}
            results[f"{model_name}_{dim}_{prompt}"]["pearson"] = pearson
            results[f"{model_name}_{dim}_{prompt}"]["spearman"] = spearman
            results[f"{model_name}_{dim}_{prompt}"]["kendall"] = kendall
            results[f"{model_name}_{dim}_{prompt}"]["mae"] = mae
            results[f"{model_name}_{dim}_{prompt}"]["rmse"] = rmse
# %%
pd.DataFrame(results).T.loc.round(3)
# %%