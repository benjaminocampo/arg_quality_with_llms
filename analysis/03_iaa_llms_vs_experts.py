# %%
import pandas as pd
import numpy as np

df = pd.read_csv("../data/webis_only_args_all_pred_llms.csv")
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
from sklearn.metrics import cohen_kappa_score

results = {}
for model_name in MODEL_NAMES:
    for dim in DIMENSIONS:
        for prompt in PROMPTS:
            run_name = f"{model_name}_{dim}_{prompt}"
            results[run_name] = {}
            results[run_name]["cohen_kappa"] = cohen_kappa_score(df[f"{run_name}_1"],
                                                                 df[f"Comparison {dim.capitalize()}al"],
                                                                 labels=["B", "Tie", "A"],
                                                                 weights="linear")
# %%
pd.DataFrame(results).T.round(3)
# %%
def get_winner(row, col):
    if row[col] == "A":
        return str(row["Argument ID A"])
    elif row[col] == "B":
        return str(row["Argument ID B"])
    else:
        return "Tie"


results = {}
for model_name in MODEL_NAMES:
    for dim in DIMENSIONS:
        for prompt in PROMPTS:
            run_name = f"{model_name}_{dim}_{prompt}"
            expert = df.apply(lambda row: get_winner(row, col=f"Comparison {dim.capitalize()}al"), axis=1)
            llm = df.apply(lambda row: get_winner(row, col=f"{run_name}_count"), axis=1)
            
            results[run_name] = {}
            results[run_name]["%_agreement_with_ids"] = (expert == llm).sum() / len(df) * 100
            results[run_name]["%_agreement_with_labels"] = (df[f"{run_name}_count"] == df[f"Comparison {dim.capitalize()}al"]).sum() / len(df) * 100
            results[run_name]["cohen_kappa_with_ids"] = cohen_kappa_score(expert, llm)
            results[run_name]["cohen_kappa_with_labels"] = cohen_kappa_score(df[f"{run_name}_count"],
                                                                             df[f"Comparison {dim.capitalize()}al"])

# %%
pd.DataFrame(results).T.round(3)
# %%
