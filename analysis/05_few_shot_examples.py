# %%
import pandas as pd

df = pd.read_csv("../data/webis_comp_orig.csv")
# %%
(
    df
    .loc[df["Logical Quality A"] > df["Logical Quality B"]]
    .sort_values(
        by=["Logical Quality A", "Logical Quality B"],
        ascending=[False, True])
    .loc[:, ["Logical Quality A", "Logical Quality B", "Comparison Logical"]]
    .tail(1)
)
# %%
dims = ["logic", "rhetoric", "dialectic"]
nof_topics = df["Topic ID A"].nunique()

examples_A = {}
for topic_id in range(1, nof_topics + 1):
    examples_A[topic_id] = {}
    for dim in dims:
        examples_A[topic_id][dim] = (
            df
            .loc[(df[f"{dim.capitalize()}al Quality A"] > df[f"{dim.capitalize()}al Quality B"]) &
                 (df[f"Comparison {dim.capitalize()}al"] == "A") &
                 (df["Topic ID A"] == topic_id)]
            .sort_values(
                by=[f"{dim.capitalize()}al Quality A",
                    f"{dim.capitalize()}al Quality B"],
                ascending=[False, True])
            .head(1)
        )
# %%
examples_A_args = {}
for topic_id in range(1, nof_topics + 1):
    examples_A_args[topic_id] = {}
    for dim in dims:
        examples_A_args[topic_id][dim] = {}
        examples_A_args[topic_id][dim]["arg A"] = examples_A[topic_id][dim]["Premise A"].iloc[0]
        examples_A_args[topic_id][dim]["arg B"] = examples_A[topic_id][dim]["Premise B"].iloc[0]
# %%
examples_B = {}
for topic_id in range(1, nof_topics + 1):
    examples_B[topic_id] = {}
    for dim in dims:
        examples_B[topic_id][dim] = (
            df
            .loc[(df[f"{dim.capitalize()}al Quality B"] > df[f"{dim.capitalize()}al Quality A"]) &
                 (df[f"Comparison {dim.capitalize()}al"] == "B") &
                 (df["Topic ID A"] == topic_id)]
            .sort_values(
                by=[f"{dim.capitalize()}al Quality B",
                    f"{dim.capitalize()}al Quality A"],
                ascending=[False, True])
            .head(1)
        )
# %%
examples_B_args = {}
for topic_id in range(1, nof_topics + 1):
    examples_B_args[topic_id] = {}
    for dim in dims:
        examples_B_args[topic_id][dim] = {}
        examples_B_args[topic_id][dim]["arg A"] = examples_B[topic_id][dim]["Premise A"].iloc[0]
        examples_B_args[topic_id][dim]["arg B"] = examples_B[topic_id][dim]["Premise B"].iloc[0]
# %%
examples_Tie = {}
for topic_id in range(1, nof_topics + 1):
    examples_Tie[topic_id] = {}
    for dim in dims:
        examples_Tie[topic_id][dim] = (
            df
            .loc[(df[f"Comparison {dim.capitalize()}al"] == "Tie") &
                 (df["Topic ID A"] == topic_id)]
            .sort_values(
                by=[f"{dim.capitalize()}al Quality B",
                    f"{dim.capitalize()}al Quality A"],
                ascending=[False, True])
            .tail(1)
        )
# %%
examples_Tie_args = {}
for topic_id in range(1, nof_topics + 1):
    examples_Tie_args[topic_id] = {}
    for dim in dims:
        examples_Tie_args[topic_id][dim] = {}
        examples_Tie_args[topic_id][dim]["arg A"] = examples_Tie[topic_id][dim]["Premise A"].iloc[0]
        examples_Tie_args[topic_id][dim]["arg B"] = examples_Tie[topic_id][dim]["Premise B"].iloc[0]
# %%
examples_A_args[1]["logic"]["arg A"]
# %%
examples_A_args
# %%
examples = {}
for topic_id in range(1, nof_topics + 1):
    examples[f"tid_{topic_id}"] = {}
    for dim in dims:
        examples[f"tid_{topic_id}"][dim] = {}
        examples[f"tid_{topic_id}"][dim]["ex_A_argA"] = examples_A_args[topic_id][dim]["arg A"]
        examples[f"tid_{topic_id}"][dim]["ex_A_argB"] = examples_A_args[topic_id][dim]["arg B"]
        examples[f"tid_{topic_id}"][dim]["ex_B_argA"] = examples_B_args[topic_id][dim]["arg A"]
        examples[f"tid_{topic_id}"][dim]["ex_B_argB"] = examples_B_args[topic_id][dim]["arg B"]
        examples[f"tid_{topic_id}"][dim]["ex_tie_argA"] = examples_Tie_args[topic_id][dim]["arg A"]
        examples[f"tid_{topic_id}"][dim]["ex_tie_argB"] = examples_Tie_args[topic_id][dim]["arg B"]
# %%
examples
# %%
