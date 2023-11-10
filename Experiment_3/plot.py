# %%
import seaborn as sns
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt

# %% [markdown]
# #### Set up

# %%
EXPERIMENT_NAMES = ["MyExperiment"]
EXPERIMENT_OPT_REWARDS = [1.1518511683670583]
METRICS = ["metrics.reward_ourAlgo", "metrics.reward_baseline2", "metrics.reward_baseline3", "metrics.reward_baseline4"]
ALGO_NAMES = ["CoBA (Ours)", "EqualAlloc", "MaxSum", "PropToValue"]
COLORS = ["blue", "green", "orange", "pink"]
PARAMS = ["params.B"]

# %%
palette = dict(zip(ALGO_NAMES, COLORS))
palette

# %%
from run_params import NUM_RUNS_PER_B, NUM_SUB_RUNS

# %%
num_runs_per_B = NUM_RUNS_PER_B
num_sub_runs = NUM_SUB_RUNS
num_stds = 2

# %%
mlflow.search_experiments()

# %%
def plot_errorbars(r):
    return (
        np.mean(r)-(num_stds*np.std(r)/np.sqrt(num_runs_per_B*num_sub_runs*len(EXPERIMENT_NAMES))),
        np.mean(r)+(num_stds*np.std(r)/np.sqrt(num_runs_per_B*num_sub_runs*len(EXPERIMENT_NAMES)))
    )

# %%
def get_regret(experiment_name, opt_reward):
    df = mlflow.search_runs(experiment_names=[experiment_name])

    df_selected = df[METRICS + PARAMS]
    df_selected = df_selected.sort_values(by=PARAMS[0])

    ## Compute rewards
    dfs = []
    for i in range(len(METRICS)):
        df_temp = {}
        df_temp = df_selected[PARAMS + [METRICS[i]]]
        df_temp.columns = ["B", "reward"]
        df_temp["algo"]=ALGO_NAMES[i]

        dfs.append(df_temp)

        df_new = pd.concat(dfs)

        df_new = df_new.sort_values(by="B")
        df_new["B"] = pd.to_numeric(df_new["B"])

    ## Compute regret
    df_new_regret = df_new[:]
    df_new_regret["reward"] = opt_reward - df_new_regret["reward"]
    df_new_regret.rename(columns={'reward' : 'regret'}, inplace=True)
    
    
    df_new_regret["B"] = pd.to_numeric(df_new_regret["B"])
    
    return df_new_regret

# %% [markdown]
# ### Get all regrets

# %%
def get_all_regrets():
    regret_df_combined = None
    for i in range(len(EXPERIMENT_NAMES)):
        # print(f"i={i}")
        # print(f"auc_df = {auc_df}")
        if regret_df_combined is None:
            regret_df_combined = get_regret(EXPERIMENT_NAMES[i], EXPERIMENT_OPT_REWARDS[i])  
        else:
            regret_df_combined = pd.concat([regret_df_combined, get_regret(EXPERIMENT_NAMES[i], EXPERIMENT_OPT_REWARDS[i])])
        

    return regret_df_combined

# %%
regret_df_combined = get_all_regrets()

# %%
marker_styles = dict(zip(ALGO_NAMES, ['^', 'o', 'o', 'o']))

# %%
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11

axes = sns.lineplot(
    regret_df_combined,
    x="B",
    y="regret",
    hue="algo",
    style="algo",
    errorbar = plot_errorbars,
    err_style = "bars",
    err_kws = {'capsize':5, 'elinewidth':0.5, 'capthick':0.5},
    palette = palette,
    markers = marker_styles,
    dashes=[(3, 2)]*4,
)

axes.set_xticks(list(regret_df_combined["B"].unique()))

axes.set_xlabel("Budget $B$")
axes.set_ylabel("Regret")

handles, labels = axes.get_legend_handles_labels()
axes.legend(handles, labels)

plt.show()


# %%
fig = axes.get_figure()
fig.savefig(f"Experiment3.png")

# %%



