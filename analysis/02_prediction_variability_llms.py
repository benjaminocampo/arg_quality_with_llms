# %%
import pandas as pd
import numpy as np

df_scores = pd.read_csv("../data/webis_only_args_all_pred_llms.csv")
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
PROMPTS = ["zero_shot", "cot"]
# %%
from collections import Counter

def majority_vote(row, col):
    values = [row[f"{col}_0"], row[f"{col}_1"], row[f"{col}_2"]]
    counts = Counter(values)
    most_common_value, count = counts.most_common(1)[0]
    if count >= 2:
        return pd.Series({col: most_common_value, f"{col}_agreement_count": count})
    else:
        return pd.Series({col: "All unequal", f"{col}_agreement_count": 1})
# %% 
import krippendorff
import numpy as np

results = {}
for model_name in MODEL_NAMES:
    for dim in DIMENSIONS:
        for prompt in PROMPTS:
            run_name = f"{model_name}_{dim}_{prompt}"

            df_variability = df_scores.apply(lambda row: majority_vote(row, run_name), axis=1)
            results[run_name] = df_variability[f"{run_name}_agreement_count"].value_counts().to_dict()
            results[run_name].setdefault(1, 0)
            results[run_name].setdefault(2, 0)
            results[run_name].setdefault(3, 0)
            results[run_name]["3_%"] = results[run_name][3] / len(df_scores) * 100
            results[run_name]["2_%"] = results[run_name][2] / len(df_scores) * 100
            results[run_name]["1_%"] = results[run_name][1] / len(df_scores) * 100

            run_1 = df_scores[f"{run_name}_0"].replace({"B": 0, "Tie": 1, "A": 2}).astype(int)
            run_2 = df_scores[f"{run_name}_1"].replace({"B": 0, "Tie": 1, "A": 2}).astype(int)
            run_3 = df_scores[f"{run_name}_2"].replace({"B": 0, "Tie": 1, "A": 2}).astype(int)
            reliability_data = np.vstack([run_1, run_2, run_3])
            results[run_name]["krip_alpha"] = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement="nominal")
# %%
df_pred_var = pd.DataFrame(results).T
# %%
df_pred_var = df_pred_var.rename(columns={3: "3", 2: "2", 1: "1"})
# %%
df_pred_var[["3", "3_%", "2", "2_%", "1", "1_%", "krip_alpha"]].round(2)
# %%
