# %%
import pandas as pd

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
df_scores
# %%
from scipy.stats import zscore

results = {}
#for model_name in MODEL_NAMES:
#    for dim in DIMENSIONS:
#        for prompt in PROMPTS:


run_name = f"{model_name}_{dim}_{prompt}_bt_zscore"
expert_z = zscore(df_scores[f"{dim.capitalize()}al Quality"])
llm_z = zscore(df_scores[run_name])
expert_score = df_scores[f"{dim.capitalize()}al Quality"]
llm_score = df_scores[run_name]
# %%
import seaborn as sns

sns.displot(data=df_scores, x="Logical Quality")
# %%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

BEST_LLM = "llama-70B-big_{dim}_few_shot"
#BEST_LLM = "mistral-7B-small_{dim}_zero_shot"

# Expert z-scores
expert_z_logic = zscore(df_scores["Logical Quality"])
expert_z_rhetoric = zscore(df_scores["Rhetorical Quality"])
expert_z_dialectic = zscore(df_scores["Dialectical Quality"])

# Model z-scores
llm_z_logic = zscore(
    df_scores[f"{BEST_LLM.format(dim='logic')}_bt_zscore"]
)
llm_z_rhetoric = zscore(
    df_scores[f"{BEST_LLM.format(dim='rhetoric')}_bt_zscore"]
)
llm_z_dialectic = zscore(
    df_scores[f"{BEST_LLM.format(dim='dialectic')}_bt_zscore"]
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

dimensions = [
    ("Logic", expert_z_logic, llm_z_logic),
    ("Rhetoric", expert_z_rhetoric, llm_z_rhetoric),
    ("Dialectic", expert_z_dialectic, llm_z_dialectic),
]

expert_color = "#1f77b4"
model_color = "#ff7f0e"

title_fontsize = 20
label_fontsize = 16
axis_label_fontsize = 18
tick_fontsize = 16

for ax, (title, expert_vals, model_vals) in zip(axes, dimensions):
    sns.kdeplot(
        expert_vals,
        ax=ax,
        fill=True,
        color=expert_color,
        label="Expert",
        alpha=0.3,
    )
    sns.kdeplot(
        model_vals,
        ax=ax,
        fill=True,
        color=model_color,
        label="Llama-70B\nfew-shot",
        alpha=0.3,
    )

    ax.set_title(title, fontsize=title_fontsize)

    ax.set_xlabel("Bradley-Terry Score", fontsize=axis_label_fontsize)
    ax.set_ylabel("Density", fontsize=axis_label_fontsize)

    ax.tick_params(axis="both", labelsize=tick_fontsize)

    ax.grid(alpha=0.2)

axes[0].legend(fontsize=14)

plt.tight_layout()
plt.tight_layout()
plt.show()
# %%
fig.savefig(
    "expert_vs_best_model_kde.pdf",
    bbox_inches="tight",
)
# %%
