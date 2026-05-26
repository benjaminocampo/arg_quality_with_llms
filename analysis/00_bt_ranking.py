# %%
import pandas as pd
import numpy as np
import choix


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
DIMENSIONS = ["logic", "rhetoric", "dialectic"]
#DIMENSIONS = ["logic", "rhetoric", "dialectic"]
PROMPTS = ["zero_shot", "few_shot", "cot"]
#PROMPTS = ["cot"]
RUN_IDS = ["zero", "one", "two"]
# %% [markdown]
# Counting system
# %%
def count_label(labels):
    # Count of Ties is basically a 0 to A and B
    labels_list = labels.tolist()
    diff = labels_list.count("A") - labels_list.count("B")
    
    if diff > 0:
        return "A"
    elif diff < 0:
        return "B"
    return "Tie"


for model_name in MODEL_NAMES:
    for dim in DIMENSIONS:
        for prompt in PROMPTS:
            df0 = pd.read_csv(f"../gens/{model_name}_{dim}_{prompt}_zero.csv")
            df1 = pd.read_csv(f"../gens/{model_name}_{dim}_{prompt}_one.csv")
            df2 = pd.read_csv(f"../gens/{model_name}_{dim}_{prompt}_two.csv")

            df_count = df0.copy()
            df_count = df_count.drop(columns=[f"Pred Comparison {dim.capitalize()}"])

            df_count[f"Pred Comparison {dim.capitalize()} 0"] = df0[f"Pred Comparison {dim.capitalize()}"]
            df_count[f"Pred Comparison {dim.capitalize()} 1"] = df1[f"Pred Comparison {dim.capitalize()}"]
            df_count[f"Pred Comparison {dim.capitalize()} 2"] = df2[f"Pred Comparison {dim.capitalize()}"]

            df_count[f"Pred Comparison {dim.capitalize()} Count"] = df_count[[f"Pred Comparison {dim.capitalize()} {rid}" for rid in [0, 1, 2]]].apply(count_label, axis=1)

            df_count.to_csv(f"../gens_aggregated/{model_name}_{dim}_{prompt}.csv", index=False)
# %%
REG_ALPHA = 0.01

for model_name in MODEL_NAMES:
    for dim in DIMENSIONS:
        for prompt in PROMPTS:
            run_name = f"{model_name}_{dim}_{prompt}"
            infile = f"../gens_aggregated/{run_name}.csv"
            outfile = f"../bt_scores/bt_score_{run_name}.csv"
            df = pd.read_csv(infile)
            df["id_a"] = df["Discussion ID A"].astype(str) + "_" + df["Argument ID A"].astype(str) + "_" + df["Topic ID A"].astype(str)
            df["id_b"] = df["Discussion ID B"].astype(str) + "_" + df["Argument ID B"].astype(str) + "_" + df["Topic ID B"].astype(str)

            # 2. Map argument IDs to 0…n−1
            all_args = pd.unique(df[["id_a", "id_b"]].values.ravel())
            arg2idx  = {arg: i for i, arg in enumerate(all_args)}
            print(len(all_args))
            comp = []
            for _, row in df.iterrows():
                a, b = arg2idx[row["id_a"]], arg2idx[row["id_b"]]
                label = str(row[f"Pred Comparison {dim.capitalize()} Count"]).strip().lower()
                if label == "a":
                    comp.append((a, b))
                elif label == "b":
                    comp.append((b, a))
                elif label == "tie":
                    comp.append((a, b))
                    comp.append((b, a))
                else:
                    print(f"Skipping invalid label: {row[f'Pred Comparison {dim.capitalize()} Count']} at row {row}")
                    continue

            # 3. Fit Bradley–Terry with ILSR
            abilities = choix.ilsr_pairwise(
                n_items=len(all_args),
                data=comp,
                alpha=REG_ALPHA
            )

            # 4. Z-normalise (μ=0, σ=1) to match Webis expert file
            abilities = (abilities - abilities.mean()) / abilities.std(ddof=0)

            # 5. Write CSV
            out_df = pd.DataFrame({
                "argument_id": all_args,
                "bt_zscore": np.round(abilities, 4)
            }).sort_values("bt_zscore", ascending=False)

            out_df.to_csv(outfile, index=False)
# %%
# Collect all scores
df_args = pd.read_csv("../data/webis_only_args.csv")
df_args["id"] = df_args["Discussion ID"].astype(str) + "_" + df_args["Argument ID"].astype(str) + "_" + df_args["Topic ID"].astype(str)
# %%
for model_name in MODEL_NAMES:
    for dim in DIMENSIONS:
        for prompt in PROMPTS:
            run_name = f"{model_name}_{dim}_{prompt}"
            infile = f"../bt_scores/bt_score_{run_name}.csv"
            df = pd.read_csv(infile)
            df_args = df_args.merge(df, left_on="id", right_on="argument_id", how="left")
            df_args = df_args.drop(columns=["argument_id"])
            df_args = df_args.rename(columns={"bt_zscore": f"{run_name}_bt_zscore"})
# %%
# %%
df_args.to_csv("../data/webis_only_args_bt_scores.csv", index=False)
# %%
df_args_all_preds = pd.read_csv("../gens_aggregated/mistral-7B-small_logic_zero_shot.csv")
df_args_all_preds = df_args_all_preds.drop(columns=["Pred Comparison Logic 0",
                                                    "Pred Comparison Logic 1",
                                                    "Pred Comparison Logic 2",
                                                    "Pred Comparison Logic Count"])
# %%
for model_name in MODEL_NAMES:
    for dim in DIMENSIONS:
        for prompt in PROMPTS:
            run_name = f"{model_name}_{dim}_{prompt}"
            infile = f"../gens_aggregated/{model_name}_{dim}_{prompt}.csv"
            df = pd.read_csv(infile)
            df_args_all_preds[f"{run_name}_0"] = df[f"Pred Comparison {dim.capitalize()} 0"]
            df_args_all_preds[f"{run_name}_1"] = df[f"Pred Comparison {dim.capitalize()} 1"]
            df_args_all_preds[f"{run_name}_2"] = df[f"Pred Comparison {dim.capitalize()} 2"]
            df_args_all_preds[f"{run_name}_count"] = df[f"Pred Comparison {dim.capitalize()} Count"]
# %%
df_args_all_preds.to_csv("../data/webis_only_args_all_pred_llms.csv", index=False)
# %%
