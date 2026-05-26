# %%
import pandas as pd
import numpy as np
from scipy.stats import zscore, pearsonr, spearmanr, kendalltau
from pathlib import Path

df_scores = pd.read_csv("../data/webis_only_args_bt_scores.csv")
# %% 
df_scores
# %%
MODEL_NAMES = [
    "mistral-7B-small",
    "llama-8B-small",
    "olmo2-7B-small",
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
PROMPTS = ["zero_shot", "few_shot", "cot"]
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
df_results_bt = pd.DataFrame(results).T
# %%
df = pd.read_csv("../data/webis_only_args_all_pred_llms.csv")
# %%
from sklearn.metrics import cohen_kappa_score

def get_winner(row, col):
    if row[col] == "A":
        return str(row["Argument ID A"])
    elif row[col] == "B":
        return str(row["Argument ID B"])
    else:
        return "Tie"


results_aggreement = {}
for model_name in MODEL_NAMES:
    for dim in DIMENSIONS:
        for prompt in PROMPTS:
            run_name = f"{model_name}_{dim}_{prompt}"
            expert = df.apply(lambda row: get_winner(row, col=f"Comparison {dim.capitalize()}al"), axis=1)
            llm = df.apply(lambda row: get_winner(row, col=f"{run_name}_count"), axis=1)
            
            results_aggreement[run_name] = {}
            results_aggreement[run_name]["%_agreement_with_ids"] = (expert == llm).sum() / len(df) * 100
            results_aggreement[run_name]["%_agreement_with_labels"] = (df[f"{run_name}_count"] == df[f"Comparison {dim.capitalize()}al"]).sum() / len(df) * 100
            results_aggreement[run_name]["cohen_kappa_with_ids"] = cohen_kappa_score(expert, llm)
            results_aggreement[run_name]["cohen_kappa_with_labels"] = cohen_kappa_score(df[f"{run_name}_count"],
                                                                                        df[f"Comparison {dim.capitalize()}al"])

# %%
df_results_agreement = pd.DataFrame(results_aggreement).T
# %%
assert all(df_results_bt.index == df_results_agreement.index)
# %%
run_names = pd.concat([df_results_bt, df_results_agreement], axis=1).round(3).index
# %%
model_names, dims, prompts = zip(*(r.split("_") for r in run_names))

model_names = list(model_names)
dims = list(dims)
prompts = list(prompts)
# %%
df_results_all = pd.concat([df_results_bt, df_results_agreement], axis=1)
# %%
df_results_all["Model"] = model_names
df_results_all["Dim"] = dims
df_results_all["Prompt"] = prompts
# %%
df_results_all_latex = (
    df_results_all
    .reset_index(drop=True)
    .drop(columns=["Dim"])
    .rename(columns={"%_agreement_with_ids": "\% Agree. w/ ids",
                     "%_agreement_with_labels": "\% Agree. w/ labels",
                     "cohen_kappa_with_ids": "\kappa w/ ids",
                     "cohen_kappa_with_labels": "\kappa w/ labels"})
    .drop(columns=["\% Agree. w/ labels", "\kappa w/ labels"])
    .groupby(["Model", "Prompt"])
    .agg(["mean", "std"])
    .round(3)
)
# %%
df_results_all_latex[[('\% Agree. w/ ids', 'mean'), ('\% Agree. w/ ids',  'std')]] = df_results_all_latex[[('\% Agree. w/ ids', 'mean'), ('\% Agree. w/ ids',  'std')]].round(2)
# %%
result_pm = pd.DataFrame(index=df_results_all_latex.index)

for col in df_results_all_latex.columns.levels[0]:
    result_pm[col] = (
        df_results_all_latex[(col, "mean")].map("{:.3f}".format)
        + " $\pm$ "
        + df_results_all_latex[(col, "std")].map("{:.3f}".format)
    )
# %%
from itertools import product
pairs = list(product(MODEL_NAMES, [p.replace("_shot", "") for p in PROMPTS]))
# %%
print(result_pm.loc[pairs][['pearson', 'spearman', 'kendall', 'mae', 'rmse', '\kappa w/ ids', '\% Agree. w/ ids']].to_latex())
# %%
result_pm.loc[pairs][['pearson', 'spearman', 'kendall', 'mae', 'rmse', '\kappa w/ ids', '\% Agree. w/ ids']]
# %%
df_results_all_with_dim = (
    df_results_all
    .reset_index(drop=True)
    .rename(columns={"%_agreement_with_ids": "\% Agree. w/ ids",
                     "%_agreement_with_labels": "\% Agree. w/ labels",
                     "cohen_kappa_with_ids": "\kappa w/ ids",
                     "cohen_kappa_with_labels": "\kappa w/ labels"})
    .drop(columns=["\% Agree. w/ labels", "\kappa w/ labels"])
)
# %%
print((
    df_results_all_with_dim
    .loc[(df_results_all_with_dim["Model"] == "llama-70B-big") &
         (df_results_all_with_dim["Prompt"] == "few"),
         ["Dim", "pearson", "spearman", "kendall", "mae", "rmse", "\% Agree. w/ ids", "\kappa w/ ids"]]
    .set_index("Dim")
    .round(3)
    .to_latex()
))
# %%
