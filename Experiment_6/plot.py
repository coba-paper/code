# %%
import seaborn as sns
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt

# %% [markdown]
# #### Set up

# %%
EXPERIMENT_NAMES = ["MyExperiment1", "MyExperiment2"]  # Provide the list of experiment names to plot (each experiment will be a tick in the x-axis)
EXPERIMENT_PARAMS = ["beta-constant", "beta-inv-prop"] 
METRICS = ["metrics.reward_ourAlgo", "metrics.reward_baseline2", "metrics.reward_baseline3", "metrics.reward_baseline4"]
ALGO_NAMES = ["CoBA (Ours)", "EqualAlloc", "MaxSum", "PropToValue"]
COLORS = ["blue", "green", "orange", "pink"]
PARAMS = ["params.B"]

# %%
palette = dict(zip(ALGO_NAMES, COLORS))

# %%
from run_params import NUM_RUNS_PER_B, NUM_SUB_RUNS

# %%
num_runs_per_B = NUM_RUNS_PER_B
num_sub_runs = NUM_SUB_RUNS
num_stds = 2

# %%
def plot_errorbars(r):
    return (
        np.mean(r)-(num_stds*np.std(r)/np.sqrt(num_runs_per_B*num_sub_runs)), 
        np.mean(r)+(num_stds*np.std(r)/np.sqrt(num_runs_per_B*num_sub_runs))
    )

# %%
def get_regret(experiment_name):
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
    df_new_regret["reward"] = 1.6 - df_new_regret["reward"]
    df_new_regret.rename(columns={'reward' : 'regret'}, inplace=True)
    
    
    df_new_regret["B"] = pd.to_numeric(df_new_regret["B"])
    
    return df_new_regret

# %% [markdown]
# Plot

# %%
def auc(x):
    
    m = x[["B", "regret"]].groupby('B').mean().sum()
    z = num_runs_per_B*num_sub_runs
    s = x[["B", "regret"]].groupby('B').std().sum()/(np.sqrt(z))

    df = pd.DataFrame({'mean':m, 'std':s})

    return df

# %%
def compute_auc(df_new_regret):
    B_values = df_new_regret["B"].unique()
    
    auc_df = df_new_regret.groupby(by=['algo']).apply(auc)    

    return auc_df
    


# %% [markdown]
# ### AUC computation

# %%
def get_all_auc():
    auc_df_combined = None
    for i in range(len(EXPERIMENT_NAMES)):
        auc_df = compute_auc(get_regret(EXPERIMENT_NAMES[i]))        
        auc_df.index = auc_df.index.get_level_values(level='algo')
        auc_df.reset_index(inplace=True)
        
        if auc_df_combined is None:
            auc_df['param'] = EXPERIMENT_PARAMS[i]
            auc_df_combined = pd.DataFrame(auc_df)
        else:
            auc_df['param'] = EXPERIMENT_PARAMS[i]
            auc_df_combined = pd.concat([auc_df_combined, pd.DataFrame(auc_df)])
        

    return auc_df_combined

# %%
final_auc = get_all_auc()



# %% [markdown]
# Plot

# %%
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11


expt_names = ['-'.join(_.split('-')[1:]) for _ in EXPERIMENT_NAMES]

fig = plt.figure()

# create data
x = np.array([0, 0.65])
y = []
for algo in ALGO_NAMES:
    y += [final_auc[final_auc['algo']==algo]['mean']]
width = 0.1

# plot data in grouped manner of bar type
for i, algo in enumerate(ALGO_NAMES):
    plt.bar(x+(0.1*(2-i)), y[i], width, color=COLORS[i])

plt.xticks(x, ['$\\beta(.) \propto N_{x, {c}^A} $', '$\\beta(.) \propto \\frac{N_{x, c^A}}{P({c^A})}$'])
plt.xlabel(f"$\\beta(.)$ function")
plt.ylabel("Regret AUC")
plt.legend(ALGO_NAMES)

for i, algo in enumerate(ALGO_NAMES):
    plt.errorbar(
        x=x+(0.1*(2-i)), 
        y=final_auc[final_auc['algo']==algo]['mean'], 
        yerr=list(final_auc[final_auc['algo']==algo]['std']),
        capsize=4,
        linestyle='',
        color="grey", 
)

plt.show()









# %%
expt_names = ['-'.join(_.split('-')[1:]) for _ in EXPERIMENT_NAMES]
fig.savefig(f"Experiment6.png", bbox_inches='tight')



