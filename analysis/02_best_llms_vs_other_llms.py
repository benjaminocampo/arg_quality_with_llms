# %%
import pandas as pd
import numpy as np
from scipy.stats import zscore, pearsonr, spearmanr, kendalltau
from pathlib import Path

df_scores = pd.read_csv("../data/webis_only_args_bt_scores.csv")
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
BEST_LLM = "llama-70B-big_{dim}_few_shot"

results_bt = {}
for model_name in MODEL_NAMES:
    for dim in DIMENSIONS:
        for prompt in PROMPTS:
            run_name = f"{model_name}_{dim}_{prompt}_bt_zscore"
            best_llm_run_name = BEST_LLM.format(dim=dim)
            best_llm_run_name = f"{best_llm_run_name}_bt_zscore"
            
            # Adding suffix: RhetoricAL, LogicAL, DialecticAL
            best_llm_z = zscore(df_scores[best_llm_run_name])
            llm_z = zscore(df_scores[run_name])
            best_llm_score = df_scores[best_llm_run_name]
            llm_score = df_scores[run_name]

            pearson  = pearsonr(best_llm_z, llm_z).statistic
            spearman = spearmanr(best_llm_score, llm_score).correlation 
            kendall  = kendalltau(best_llm_score, llm_score).correlation  
            mae      = np.mean(np.abs(best_llm_z - llm_z))
            rmse     = np.sqrt(np.mean((best_llm_z - llm_z)**2))

            results_bt[f"{model_name}_{dim}_{prompt}"] = {}
            results_bt[f"{model_name}_{dim}_{prompt}"]["pearson"] = pearson
            results_bt[f"{model_name}_{dim}_{prompt}"]["spearman"] = spearman
            results_bt[f"{model_name}_{dim}_{prompt}"]["kendall"] = kendall
            results_bt[f"{model_name}_{dim}_{prompt}"]["mae"] = mae
            results_bt[f"{model_name}_{dim}_{prompt}"]["rmse"] = rmse
# %%
df_results_bt = pd.DataFrame(results_bt).T
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


results_agreement = {}
for model_name in MODEL_NAMES:
    for dim in DIMENSIONS:
        for prompt in PROMPTS:
            run_name = f"{model_name}_{dim}_{prompt}"
            llm_best = df.apply(lambda row: get_winner(row, col=f"{BEST_LLM.format(dim=dim)}_count"), axis=1)
            llm = df.apply(lambda row: get_winner(row, col=f"{run_name}_count"), axis=1)
            
            results_agreement[run_name] = {}
            results_agreement[run_name]["%_agreement_with_ids"] = (llm_best == llm).sum() / len(df) * 100
            results_agreement[run_name]["%_agreement_with_labels"] = (df[f"{run_name}_count"] == df[f"{BEST_LLM.format(dim=dim)}_count"]).sum() / len(df) * 100
            results_agreement[run_name]["cohen_kappa_with_ids"] = cohen_kappa_score(llm_best, llm)
            results_agreement[run_name]["cohen_kappa_with_labels"] = cohen_kappa_score(df[f"{run_name}_count"],
                                                                                       df[f"{BEST_LLM.format(dim=dim)}_count"])
# %%
df_results_agreement = pd.DataFrame(results_agreement).T
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
