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
PROMPTS = ["zero_shot", "few_shot", "cot"]
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
df_pred_var
# %%
model_names, dims, prompts = zip(*(r.split("_") for r in df_pred_var.index))

model_names = list(model_names)
dims = list(dims)
prompts = list(prompts)
# %%
# df_results_all = pd.concat([df_results_bt, df_results_agreement], axis=1)
# %%
df_pred_var["Model"] = model_names
df_pred_var["Dim"] = dims
df_pred_var["Prompt"] = prompts
# %%
df_pred_var_mean_std = (
    df_pred_var
    .reset_index(drop=True)
    .drop(columns=["Dim"])
    .groupby(["Model", "Prompt"])
    .agg(["mean", "std"])
    .round(2)
    .loc[:, ["3", "3_%", "2", "2_%", "1", "1_%", "krip_alpha"]]
)
# %%
from itertools import product
pairs = list(product(MODEL_NAMES, [p.replace("_shot", "") for p in PROMPTS]))
# %%
df_results_pm = pd.DataFrame(index=df_pred_var_mean_std.index)

for col in df_pred_var_mean_std.columns.levels[0]:
    df_results_pm[col] = (
        df_pred_var_mean_std[(col, "mean")].map("{:.2f}".format)
        + " $\pm$ "
        + df_pred_var_mean_std[(col, "std")].map("{:.2f}".format)
    )
# %%
print(df_results_pm.loc[pairs, ["3", "3_%", "2", "2_%", "1", "1_%", "krip_alpha"]].to_latex())
# %%
