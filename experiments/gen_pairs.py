# %%
import pandas as pd
from utils import group_args_by_topic_with_indices, create_cyclic_pairs_within_topics

df = pd.read_csv("../data/webis_only_args.csv")
# %%
df.columns = df.columns.str.strip()
df.rename(columns={
    "Topic ID": "topic_id",
    "Discussion ID": "discussion_id",
    "Argument ID": "argument_id",
    "Premise": "premise"
}, inplace=True)    
args = df.to_dict(orient="records")

# Group arguments by topic and generate cyclic pairs within topics
grouped_args = group_args_by_topic_with_indices(args)
pairs = create_cyclic_pairs_within_topics(grouped_args, step=5)
# %%
pairs
# %%
comp = {}
comp['Arg A'] = []
comp['Arg B'] = []
comp['Discussion ID A'] = []
comp['Argument ID A'] = []
comp['Discussion ID B'] = []
comp['Argument ID B'] = []
comp['Topic ID'] = []
comp["Rhetorical Quality A"] = []
comp["Logical Quality A"] = []
comp["Dialectical Quality A"] = []
comp["Combined Quality A"] = []
comp["Rhetorical Quality B"] = []
comp["Logical Quality B"] = []
comp["Dialectical Quality B"] = []
comp["Combined Quality B"] = []

for _, (i, j) in enumerate(pairs):
    a = args[i]
    b = args[j]
    comp['Discussion ID A'].append(a['discussion_id'])
    comp['Argument ID A'].append(a['argument_id'])
    comp['Discussion ID B'].append(b['discussion_id'])
    comp['Argument ID B'].append(b['argument_id'])
    comp['Topic ID'].append(a['topic_id'])
    comp["Arg A"].append(a["premise"])
    comp["Arg B"].append(b["premise"])
    comp["Rhetorical Quality A"].append(a["Rhetorical Quality"])
    comp["Logical Quality A"].append(a["Logical Quality"])
    comp["Dialectical Quality A"].append(a["Dialectical Quality"])
    comp["Combined Quality A"].append(a["Combined Quality"])
    comp["Rhetorical Quality B"].append(b["Rhetorical Quality"])
    comp["Logical Quality B"].append(b["Logical Quality"])
    comp["Dialectical Quality B"].append(b["Dialectical Quality"])
    comp["Combined Quality B"].append(a["Combined Quality"])

# %%
pd.DataFrame(comp).to_csv("../data/webis_comp_step5.csv", index=False)
# %%
df_comp = pd.DataFrame(comp)
# %%
df_comp[df_comp["Logical Quality A"] == df_comp["Logical Quality B"]]
# %% [markdown]
# ## Preprocessing using original comparisons from Webis
# %%
import pandas as pd
df = pd.read_csv("../data/webis_only_args.csv")
df_comp_dial = pd.read_csv("../data/webis-argquality20-pairwise-dialectical.csv")
# %%
df_comp_dial_mg = df_comp_dial.merge(df[["Argument ID", "Discussion ID", "Topic ID"]],
                                     left_on=["Argument ID A", "Discussion ID A"],
                                     right_on=["Argument ID", "Discussion ID"],
                                     how="left")
# %%
df_comp_dial_mg = (
    df_comp_dial_mg
    .drop(columns=["Argument ID", "Discussion ID"])
    .rename(columns={"Topic ID": "Topic ID A"})
)
# %%
df_comp_dial_mg_all = df_comp_dial_mg.merge(df[["Argument ID", "Discussion ID", "Topic ID"]],
                                            left_on=["Argument ID B", "Discussion ID B"],
                                            right_on=["Argument ID", "Discussion ID"],
                                            how="left")
# %%
df_comp_dial_mg_all = (
    df_comp_dial_mg_all
    .drop(columns=["Argument ID", "Discussion ID"])
    .rename(columns={"Topic ID": "Topic ID B"})
)
# %%
df_comp_dial_mg_all = df_comp_dial_mg_all[df_comp_dial_mg_all["Topic ID A"] == df_comp_dial_mg_all["Topic ID B"]]
# %%
df_comp_dial_mg_all = df_comp_dial_mg_all.reset_index(drop=True)
# %%
assert all(df_comp_dial_mg_all[["Discussion ID A", "Argument ID A"]] == df_comp_dial[["Discussion ID A", "Argument ID A"]])
# %%
assert all(df_comp_dial_mg_all[["Discussion ID B", "Argument ID B"]] == df_comp_dial[["Discussion ID B", "Argument ID B"]])
# %%
df_comp_dial_mg_all = (
    df_comp_dial_mg_all
    .merge(df,
           left_on=["Argument ID A", "Discussion ID A", "Topic ID A"],
           right_on=["Argument ID", "Discussion ID", "Topic ID"])
    .drop(columns=["Argument ID", "Discussion ID", "Topic ID", "Is Argument?"])
    .rename(columns={"Premise": "Premise A",
                     "Relevance": "Relevance A",
                     "Rhetorical Quality": "Rhetorical Quality A",
                     "Logical Quality": "Logical Quality A",
                     "Dialectical Quality": "Dialectical Quality A",
                     "Text Length": "Text Length A",
                     "Stance": "Stance A",
                     "Combined Quality": "Combined Quality A"})
    .merge(df,
           left_on=["Argument ID B", "Discussion ID B", "Topic ID B"],
           right_on=["Argument ID", "Discussion ID", "Topic ID"])
    .drop(columns=["Argument ID", "Discussion ID", "Topic ID", "Is Argument?"])
    .rename(columns={"Premise": "Premise B",
                     "Relevance": "Relevance B",
                     "Rhetorical Quality": "Rhetorical Quality B",
                     "Logical Quality": "Logical Quality B",
                     "Dialectical Quality": "Dialectical Quality B",
                     "Text Length": "Text Length B",
                     "Stance": "Stance B",
                     "Combined Quality": "Combined Quality B"})
)
# %%
df_comp_dial_mg_all = df_comp_dial_mg_all.rename(columns={"Comparison": "Comparison Dialectical"})
# %%
df_comp_log = pd.read_csv("../data/webis-argquality20-pairwise-logical.csv")
# %%
df_comp_log_mg = df_comp_log.merge(df[["Argument ID", "Discussion ID", "Topic ID"]],
                                     left_on=["Argument ID A", "Discussion ID A"],
                                     right_on=["Argument ID", "Discussion ID"],
                                     how="left")
# %%
df_comp_log_mg = (
    df_comp_log_mg
    .drop(columns=["Argument ID", "Discussion ID"])
    .rename(columns={"Topic ID": "Topic ID A"})
)
# %%
df_comp_log_mg_all = df_comp_log_mg.merge(df[["Argument ID", "Discussion ID", "Topic ID"]],
                                            left_on=["Argument ID B", "Discussion ID B"],
                                            right_on=["Argument ID", "Discussion ID"],
                                            how="left")
# %%
df_comp_log_mg_all = (
    df_comp_log_mg_all
    .drop(columns=["Argument ID", "Discussion ID"])
    .rename(columns={"Topic ID": "Topic ID B"})
)
# %%
df_comp_log_mg_all = df_comp_log_mg_all[df_comp_log_mg_all["Topic ID A"] == df_comp_log_mg_all["Topic ID B"]]
# %%
df_comp_log_mg_all = df_comp_log_mg_all.reset_index(drop=True)
# %%
assert all(df_comp_log_mg_all[["Discussion ID A", "Argument ID A"]] == df_comp_log[["Discussion ID A", "Argument ID A"]])
# %%
assert all(df_comp_log_mg_all[["Discussion ID B", "Argument ID B"]] == df_comp_log[["Discussion ID B", "Argument ID B"]])
# %%
df_comp_log_mg_all = (
    df_comp_log_mg_all
    .merge(df,
           left_on=["Argument ID A", "Discussion ID A", "Topic ID A"],
           right_on=["Argument ID", "Discussion ID", "Topic ID"])
    .drop(columns=["Argument ID", "Discussion ID", "Topic ID", "Is Argument?"])
    .rename(columns={"Premise": "Premise A",
                     "Relevance": "Relevance A",
                     "Rhetorical Quality": "Rhetorical Quality A",
                     "Logical Quality": "Logical Quality A",
                     "Dialectical Quality": "Dialectical Quality A",
                     "Text Length": "Text Length A",
                     "Stance": "Stance A",
                     "Combined Quality": "Combined Quality A"})
    .merge(df,
           left_on=["Argument ID B", "Discussion ID B", "Topic ID B"],
           right_on=["Argument ID", "Discussion ID", "Topic ID"])
    .drop(columns=["Argument ID", "Discussion ID", "Topic ID", "Is Argument?"])
    .rename(columns={"Premise": "Premise B",
                     "Relevance": "Relevance B",
                     "Rhetorical Quality": "Rhetorical Quality B",
                     "Logical Quality": "Logical Quality B",
                     "Dialectical Quality": "Dialectical Quality B",
                     "Text Length": "Text Length B",
                     "Stance": "Stance B",
                     "Combined Quality": "Combined Quality B"})
)
# %%
df_comp_log_mg_all = df_comp_log_mg_all.rename(columns={"Comparison": "Comparison Logical"})
# %%
df_comp_log_mg_all
# %%
df_comp_rhet = pd.read_csv("../data/webis-argquality20-pairwise-rhetorical.csv")
# %%
df_comp_rhet_mg = df_comp_rhet.merge(df[["Argument ID", "Discussion ID", "Topic ID"]],
                                     left_on=["Argument ID A", "Discussion ID A"],
                                     right_on=["Argument ID", "Discussion ID"],
                                     how="left")
# %%
df_comp_rhet_mg = (
    df_comp_rhet_mg
    .drop(columns=["Argument ID", "Discussion ID"])
    .rename(columns={"Topic ID": "Topic ID A"})
)
# %%
df_comp_rhet_mg_all = df_comp_rhet_mg.merge(df[["Argument ID", "Discussion ID", "Topic ID"]],
                                            left_on=["Argument ID B", "Discussion ID B"],
                                            right_on=["Argument ID", "Discussion ID"],
                                            how="left")
# %%
df_comp_rhet_mg_all = (
    df_comp_rhet_mg_all
    .drop(columns=["Argument ID", "Discussion ID"])
    .rename(columns={"Topic ID": "Topic ID B"})
)
# %%
df_comp_rhet_mg_all = df_comp_rhet_mg_all[df_comp_rhet_mg_all["Topic ID A"] == df_comp_rhet_mg_all["Topic ID B"]]
# %%
df_comp_rhet_mg_all = df_comp_rhet_mg_all.reset_index(drop=True)
# %%
assert all(df_comp_rhet_mg_all[["Discussion ID A", "Argument ID A"]] == df_comp_rhet[["Discussion ID A", "Argument ID A"]])
# %%
assert all(df_comp_rhet_mg_all[["Discussion ID B", "Argument ID B"]] == df_comp_rhet[["Discussion ID B", "Argument ID B"]])
# %%
df_comp_rhet_mg_all = (
    df_comp_rhet_mg_all
    .merge(df,
           left_on=["Argument ID A", "Discussion ID A", "Topic ID A"],
           right_on=["Argument ID", "Discussion ID", "Topic ID"])
    .drop(columns=["Argument ID", "Discussion ID", "Topic ID", "Is Argument?"])
    .rename(columns={"Premise": "Premise A",
                     "Relevance": "Relevance A",
                     "Rhetorical Quality": "Rhetorical Quality A",
                     "Logical Quality": "Logical Quality A",
                     "Dialectical Quality": "Dialectical Quality A",
                     "Text Length": "Text Length A",
                     "Stance": "Stance A",
                     "Combined Quality": "Combined Quality A"})
    .merge(df,
           left_on=["Argument ID B", "Discussion ID B", "Topic ID B"],
           right_on=["Argument ID", "Discussion ID", "Topic ID"])
    .drop(columns=["Argument ID", "Discussion ID", "Topic ID", "Is Argument?"])
    .rename(columns={"Premise": "Premise B",
                     "Relevance": "Relevance B",
                     "Rhetorical Quality": "Rhetorical Quality B",
                     "Logical Quality": "Logical Quality B",
                     "Dialectical Quality": "Dialectical Quality B",
                     "Text Length": "Text Length B",
                     "Stance": "Stance B",
                     "Combined Quality": "Combined Quality B"})
)
# %%
df_comp_rhet_mg_all = df_comp_rhet_mg_all.rename(columns={"Comparison": "Comparison Rhetorical"})
# %%
df_comp_dial_mg_all = df_comp_dial_mg_all.sort_values(by=["Discussion ID A", "Argument ID A", "Topic ID A", "Discussion ID B", "Argument ID B", "Topic ID B"])
df_comp_dial_mg_all = df_comp_dial_mg_all.reset_index(drop=True)
# %%
df_comp_log_mg_all = df_comp_log_mg_all.sort_values(by=["Discussion ID A", "Argument ID A", "Topic ID A", "Discussion ID B", "Argument ID B", "Topic ID B"])
df_comp_log_mg_all = df_comp_log_mg_all.reset_index(drop=True)
# %%
df_comp_rhet_mg_all = df_comp_rhet_mg_all.sort_values(by=["Discussion ID A", "Argument ID A", "Topic ID A", "Discussion ID B", "Argument ID B", "Topic ID B"])
df_comp_rhet_mg_all = df_comp_rhet_mg_all.reset_index(drop=True)
# %%
assert all(
    df_comp_dial_mg_all[["Discussion ID A",
                         "Argument ID A",
                         "Topic ID A",
                         "Premise A",
                         "Relevance A",
                         "Rhetorical Quality A",
                         "Logical Quality A",
                         "Dialectical Quality A",
                         "Text Length A",
                         "Stance A",
                         "Combined Quality A",
                         "Discussion ID B",
                         "Argument ID B",
                         "Topic ID B",
                         "Premise B",
                         "Relevance B",
                         "Rhetorical Quality B",
                         "Logical Quality B",
                         "Dialectical Quality B",
                         "Text Length B",
                         "Stance B",
                         "Combined Quality B"]] ==
    df_comp_log_mg_all[["Discussion ID A",
                         "Argument ID A",
                         "Topic ID A",
                         "Premise A",
                         "Relevance A",
                         "Rhetorical Quality A",
                         "Logical Quality A",
                         "Dialectical Quality A",
                         "Text Length A",
                         "Stance A",
                         "Combined Quality A",
                         "Discussion ID B",
                         "Argument ID B",
                         "Topic ID B",
                         "Premise B",
                         "Relevance B",
                         "Rhetorical Quality B",
                         "Logical Quality B",
                         "Dialectical Quality B",
                         "Text Length B",
                         "Stance B",
                         "Combined Quality B"]]
)
# %%
assert all(
    df_comp_log_mg_all[["Discussion ID A",
                         "Argument ID A",
                         "Topic ID A",
                         "Premise A",
                         "Relevance A",
                         "Rhetorical Quality A",
                         "Logical Quality A",
                         "Dialectical Quality A",
                         "Text Length A",
                         "Stance A",
                         "Combined Quality A",
                         "Discussion ID B",
                         "Argument ID B",
                         "Topic ID B",
                         "Premise B",
                         "Relevance B",
                         "Rhetorical Quality B",
                         "Logical Quality B",
                         "Dialectical Quality B",
                         "Text Length B",
                         "Stance B",
                         "Combined Quality B"]] ==
    df_comp_rhet_mg_all[["Discussion ID A",
                         "Argument ID A",
                         "Topic ID A",
                         "Premise A",
                         "Relevance A",
                         "Rhetorical Quality A",
                         "Logical Quality A",
                         "Dialectical Quality A",
                         "Text Length A",
                         "Stance A",
                         "Combined Quality A",
                         "Discussion ID B",
                         "Argument ID B",
                         "Topic ID B",
                         "Premise B",
                         "Relevance B",
                         "Rhetorical Quality B",
                         "Logical Quality B",
                         "Dialectical Quality B",
                         "Text Length B",
                         "Stance B",
                         "Combined Quality B"]]
)
# %%
df_comp = df_comp_dial_mg_all[["Discussion ID A",
                                "Argument ID A",
                                "Topic ID A",
                                "Premise A",
                                "Discussion ID B",
                                "Argument ID B",
                                "Topic ID B",
                                "Premise B",
                                "Relevance A",
                                "Relevance B",
                                "Rhetorical Quality A",
                                "Rhetorical Quality B",
                                "Logical Quality A",
                                "Logical Quality B",
                                "Dialectical Quality A",
                                "Dialectical Quality B",
                                "Combined Quality A",
                                "Combined Quality B",
                                "Text Length A",
                                "Text Length B",
                                "Stance A",
                                "Stance B",
                                "Comparison Dialectical"]]
# %%
df_comp = pd.concat([df_comp,
                     df_comp_log_mg_all[["Comparison Logical"]],
                     df_comp_rhet_mg_all[["Comparison Rhetorical"]]], axis=1)
# %%
df_comp.to_csv("../data/webis_comp_orig.csv", index=False)
# %%
