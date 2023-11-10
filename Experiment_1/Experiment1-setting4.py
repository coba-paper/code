# %%
import numpy as np


from misc_utils import cartesian
from stat_utils_4 import get_distrib_params, ent, ent_new, P_hat, E_hat_do_x_cA, E_true_do_x_cA
from graph import Node, Graph, CPD, CPDCollection

import pprint

from stat_utils_4 import get_V_index

# %%
global prob_dict
prob_dict = None

# %% [markdown]
# ## `mlflow` and other setup

# %%
## mflow related flags

MLFLOW_ENABLED = False # Flag for whether mlflow logging needs to happen
accrue_to_specific_experiment = True # If false, then accrues to default mlflow experiment

# %%
from env_setting_4 import num_value_c1, num_value_c0, num_value_x, domain_y

# %%
import mlflow

if MLFLOW_ENABLED:
    if mlflow.active_run():
        mlflow.end_run()

    if accrue_to_specific_experiment:
        experiment_name = 'MyExperiment' # Set if needed
        tags = {
            "key": "value", 
        }

        if not mlflow.search_experiments(filter_string=f"name='{experiment_name}'"):
            experiment_id = mlflow.create_experiment(experiment_name, tags=tags)
        mlflow.set_experiment(experiment_name=experiment_name)

        print(f"Using experiment name: {experiment_name}")

    else:
        print(f"Using default experiment")


    mlflow.start_run()


# %% [markdown]
# ## Building blocks

# %% [markdown]
# ### Create Graph

# %%
# Create empty graph
g = Graph ()

# %%
# Insert graph nodes (and edges); order of parents will be retained throughout the code
g.insert('C1', [], np.arange(num_value_c1), cat='CA')
g.insert('C0', ['C1'], np.arange(num_value_c0), cat='CB')
g.insert('X', ['C0'], np.arange(num_value_x))
g.insert('Y', ['X', 'C0'], np.array(domain_y))


# %% [markdown]
# ### Create CPD collection that holds all CPDs

# %%
cpds = CPDCollection(g)
cpds

# %% [markdown]
# ### Q, v & beta functions

# %%
def get_N_index(ids, values):
    '''
        Return index in N array
    '''
    N_indices = ['X'] + cpds.g.get_list_of_CA_ids() # ['X', 'C2', 'C1']

    answer = 0

    for j in list(range(len(N_indices))):
        W = N_indices[j]
        i = ids.index(W)
        answer = answer * len(g.get_node_by_id(W).values) + values[i]

    return answer
    

# %% [markdown]
# `Q()`

# %%
def Q(id_V, pa_V, N):
  ''' Implements the Q(.) function
      id_V identifies the V in P(V|pa_V)
      pa_V specifies pa_V in P(V|pa_V)
  '''

  CA_ids = cpds.g.get_list_of_CA_ids()
  
  parents = cpds.g.get_node_by_id(id_V).parents
  parent_ids = [pa.id for pa in parents]
  ids_of_interest = [V for V in ['X']+cpds.g.get_list_of_CA_ids() if V not in parent_ids]
  possible_values = cartesian([cpds.g.get_node_by_id(id_V=_).values for _ in ids_of_interest])

  total_ent = 0    

  for v in possible_values:
    idx = get_N_index(
              ids = parent_ids+ids_of_interest,
              values=pa_V+list(v)
          )

    temp_N = N[idx]

    s = [str(_) for _ in pa_V]
    
    total_ent += (1/(1 + np.log( temp_N+1 ))) * ent_new(cpds.cpds[f"{id_V}_[{' '.join(s)}]"].beliefs)


    
  return total_ent

# %%
def get_node_values_constrained(id_V, x, cA, cB, cpds):
  '''Select and return value of V from x, cA, cB values'''

  n = cpds.g.get_node_by_id(id_V)
 
  if n.id == 'X':
    return x

  elif n.id in cpds.g.get_list_of_CA_ids():
    return cA[cpds.g.get_list_of_CA_ids().index(n.id)]

  elif n.id in cpds.g.get_list_of_CB_ids():
    return cB[cpds.g.get_list_of_CB_ids().index(n.id)]

# %% [markdown]
# `v()`

# %%
from stat_utils_4 import get_parent_values_constrained

def v_fn(N):
  ''' Implements the v(.) function '''

  answer = 0

  CA_ids = cpds.g.get_list_of_CA_ids()
  CB_ids = cpds.g.get_list_of_CB_ids()

  x_and_CA_possible_values = cartesian([cpds.g.get_node_by_id(id_V=_).values for _ in ['X']+CA_ids])
  CB_possible_values = cartesian([cpds.g.get_node_by_id(id_V=_).values for _ in CB_ids])

  for x_and_cA in x_and_CA_possible_values:
    for cB in CB_possible_values:
      sum_Q = 0
      for V in ['Y']+CB_ids:
        if V!='Y':
          sum_Q += Q(
            id_V=V, 
            pa_V=get_parent_values_constrained(id_V=V, cA=x_and_cA[1:], cB=cB, cpds=cpds), 
            N=N
          )
        else:
          sum_Q += Q(
            id_V=V, 
            pa_V=[x_and_cA[0]] + get_parent_values_constrained(id_V=V, cA=x_and_cA[1:], cB=cB, cpds=cpds), 
            N=N
          )
  
      p_hat_c = 1
      for V in CA_ids + CB_ids:
        p_hat_c *= P_hat(
          id_V=V, 
          v=get_node_values_constrained(id_V=V, x=x_and_cA[0], cA=x_and_cA[1:], cB=cB, cpds=cpds), 
          pa_V=get_parent_values_constrained(id_V=V, cA=x_and_cA[1:], cB=cB, cpds=cpds), 
          cpds=cpds
        )


      e_hat_do_x_c = 0
      for y in g.get_node_by_id('Y').values:
        e_hat_do_x_c += y * P_hat(
          id_V='Y', 
          v=y, 
          pa_V=[x_and_cA[0]]+get_parent_values_constrained(id_V='Y', cA=x_and_cA[1:], cB=cB, cpds=cpds), 
          cpds=cpds
        )

      answer += sum_Q * p_hat_c * e_hat_do_x_c

  return answer



# %% [markdown]
# `\beta()`

# %%
def beta(x, cA, N):
  ''' Implements the cost function beta(.) '''
  
  global prob_dict
  
  return N[get_N_index(ids=['X']+cpds.g.get_list_of_CA_ids(), values=[x]+cA)] 

  
  

# %% [markdown]
# ### True probability sampling functions

# %%
import math

# %%
def sample_true_prob():
    ''' 
        Sample true underlying probability distribution 
        Returns prob_dict containing the sampled distribution
    '''

    prob_dict = {}


    prob_dict["C1_[]"] = [1/num_value_c1]*num_value_c1

    for c1 in range(num_value_c1):
        prob_dict[f"C0_[{c1}]"] = [1, 0] if c1<2 else [0, 1]

    for c0 in range(num_value_c0):
        prob_dict[f"X_[{c0}]"] = [1/num_value_x]*num_value_x

    for x in range(num_value_x):
        for c0 in range(num_value_c0):           
            if c0==0:
                if x==0 or x==1:                    
                    prob_dict[f"Y_[{x} {c0}]"] = [0, 1, 0]
                elif x==2:
                    prob_dict[f"Y_[{x} {c0}]"] = [0, 0, 1]
                else:
                    prob_dict[f"Y_[{x} {c0}]"] = [1, 0, 0]
            else:
                prob_dict[f"Y_[{x} {c0}]"] = [0, 0, 1] if x==3 else [1, 0, 0]

         
    return prob_dict


# %%
def sanity_check_prob_dict():
    ''' Sanity check whether sampled probability distribution meets criteria such as marginals adding to 1'''

    prob_dict = sample_true_prob()
    CA_ids = cpds.g.get_list_of_CA_ids()
    CB_ids = cpds.g.get_list_of_CB_ids()
    ids = ['X'] + CA_ids + CB_ids + ['Y']

    for id_V in ids:
        parent_ids = [_.id for _ in cpds.g.get_node_by_id(id_V).parents]

        pa_values = cartesian([cpds.g.get_node_by_id(id_V=_).values for _ in parent_ids])
        for pa_V in pa_values:

            s = [str(_) for _ in pa_V]
            if np.sum(prob_dict[f"{id_V}_[{' '.join(s)}]"]) != 1:
                return False

    return True

# %%
assert sanity_check_prob_dict()==True

# %%
def get_true_prob(id_V, pa_V, prob_dict):
    '''Utility function to index into prob_dict'''
    
    s = [str(_) for _ in pa_V]
    return prob_dict[f"{id_V}_[{' '.join(s)}]"]

# %%
def warm_start(num_init_samples, prob_dict):
    ''' Create initial samples D_L using prob_dict '''
    
    global cpds
    samples = []

    for i in range(num_init_samples):
        sample = {}
        
        top_order = cpds.g.get_topological_order()
        cA, cB = [], []  
        CA_ids, CB_ids = [], []
        x=None
        for V in top_order:
            
            # Sample
            if V.id!='Y':
                v = np.random.choice(
                    a = g.get_node_by_id(V.id).values,
                    p = get_true_prob(
                        id_V=V.id, 
                        pa_V=get_parent_values_constrained(id_V=V.id, cA=cA, cB=cB, cpds=cpds, CA_ids=CA_ids, CB_ids=CB_ids), 
                        prob_dict=prob_dict
                    )
                )
            else:
                v = np.random.choice(
                    a = g.get_node_by_id(V.id).values,
                    p = get_true_prob(
                        id_V=V.id, 
                        pa_V=[x]+get_parent_values_constrained(id_V=V.id, cA=cA, cB=cB, cpds=cpds, CA_ids=CA_ids, CB_ids=CB_ids), 
                        prob_dict=prob_dict
                    )
                )

            sample[V.id] = v

            if V.id!='Y':
                s = [str(_) for _ in get_parent_values_constrained(id_V=V.id, cA=cA, cB=cB, cpds=cpds, CA_ids=CA_ids, CB_ids=CB_ids)]
            else:
                s = [str(_) for _ in [x]+get_parent_values_constrained(id_V=V.id, cA=cA, cB=cB, cpds=cpds, CA_ids=CA_ids, CB_ids=CB_ids)]

            cpds.cpds[f"{V.id}_[{' '.join(s)}]"].beliefs[get_V_index(V.id, v.item())] += 1

            if V.id in cpds.g.get_list_of_CA_ids():
                cA += [v]
                CA_ids += [V.id]
            elif V.id in cpds.g.get_list_of_CB_ids():
                cB += [v]
                CB_ids += [V.id]
            elif V.id == 'X':
                x = v
            

        samples += [sample]

    return samples
    


# %% [markdown]
# ## Function definitions

# %% [markdown]
# ### Optimization

# %%
def con(N):
  ''' Implements constraint for optimization'''

  beta_sum = 0

  CA_ids = cpds.g.get_list_of_CA_ids()
  
  x_and_CA_possible_values = cartesian([cpds.g.get_node_by_id(id_V=_).values for _ in ['X']+CA_ids])
  
  for x_and_cA in x_and_CA_possible_values:
    beta_sum += beta(
      x=x_and_cA[0],
      cA=list(x_and_cA[1:]),
      N=N,
    )

  return beta_sum, sum(N)

# %%
def fun(N):
  ''' Objective function for optimization'''
  return v_fn(N)

# %%
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
import numpy as np


# %%
from scipy.optimize import Bounds


# %%
Nfeval = 1

def callbackF(Xi, convergence):
    global Nfeval
    print(f"{Nfeval}  {Xi}  {fun(Xi):.3f}   {con(Xi)}") 
    Nfeval += 1



# %%
from scipy.optimize import differential_evolution

# %%
def get_integer_solutions(N):
  return [int(np.floor(n)) for n in N]

# %%
def con_scalar(N_scalar):
  ''' Implements scalar constraint for optimization for certain baselines '''
  
  CA_ids = cpds.g.get_list_of_CA_ids()
  
  x_and_CA_possible_values = cartesian([cpds.g.get_node_by_id(id_V=_).values for _ in ['X']+CA_ids])

  N = np.repeat(N_scalar, len(x_and_CA_possible_values))

  return con(N)

# %%
def fun_scalar(N_scalar):
  '''Implements scalar objective function for optimization, for use with certain baselines'''
  
  CA_ids = cpds.g.get_list_of_CA_ids()
  
  x_and_CA_possible_values = cartesian([cpds.g.get_node_by_id(id_V=_).values for _ in ['X']+CA_ids])

  N = np.repeat(N_scalar, len(x_and_CA_possible_values))

  return v_fn(N)

# %%
def callbackF_scalar(Xi):
    global Nfeval
    print("{0:4d}   {1: 3.6f}   {2: 3.6f}".format(Nfeval, Xi[0], fun_scalar(Xi)))
    Nfeval += 1



# %% [markdown]
# ### Evaluation

# %% [markdown]
# `get_samples_and_update_beliefs`

# %%
def get_samples_and_update_beliefs(N, g, cpds, prob_dict):
  ''' 
    Sample from environment 
    Number of samples given by N
    Underlying probability distribution is given by prob_dict
  '''

  samples = {}

  cpds_new = cpds.copy()

  top_order = cpds.g.get_topological_order()
  top_order_without_x_and_cA = [_ for _ in top_order if _.id not in ['X']+g.get_list_of_CA_ids()]
  
  CA_ids = cpds.g.get_list_of_CA_ids()
  
  x_and_CA_possible_values = cartesian([cpds.g.get_node_by_id(id_V=_).values for _ in ['X']+CA_ids])
  
  for x_and_cA in x_and_CA_possible_values:
    s = [str(_) for _ in x_and_cA]
    if f"{' '.join(s)}" not in samples:
      samples[f"{' '.join(s)}"] = {}
    for i in range(N[get_N_index(ids=['X']+CA_ids, values=x_and_cA)]):
      # create sub-dict for (x, cA)
      

      cB = []

      for V in top_order_without_x_and_cA:
        
        # Sample
        if V.id!='Y':
            v = np.random.choice(
                a = g.get_node_by_id(V.id).values,
                p = get_true_prob(
                    id_V=V.id, 
                    pa_V=get_parent_values_constrained(id_V=V.id, cA=x_and_cA[1:], cB=cB, cpds=cpds), 
                    prob_dict=prob_dict
                )
            )
        else:
            v = np.random.choice(
                a = g.get_node_by_id(V.id).values,
                p = get_true_prob(
                    id_V=V.id, 
                    pa_V=[x_and_cA[0]]+get_parent_values_constrained(id_V=V.id, cA=x_and_cA[1:], cB=cB, cpds=cpds), 
                    prob_dict=prob_dict
                )
            )
          
        if V.id in cpds.g.get_list_of_CB_ids():
          cB += [v]

        # Store
        if V.id not in samples[f"{' '.join(s)}"]:
          samples[f"{' '.join(s)}"][f"{V.id}"] = np.array([v])
        else:
          samples[f"{' '.join(s)}"][f"{V.id}"] = np.append(samples[f"{' '.join(s)}"][f"{V.id}"], v)

        # Update beliefs
        if V.id!='Y':
            s2 = [str(_) for _ in get_parent_values_constrained(id_V=V.id, cA=x_and_cA[1:], cB=cB, cpds=cpds)]
        else:
            s2 = [str(_) for _ in [x_and_cA[0]]+get_parent_values_constrained(id_V=V.id, cA=x_and_cA[1:], cB=cB, cpds=cpds)]

        cpds_new.cpds[f"{V.id}_[{' '.join(s2)}]"].beliefs[get_V_index(V.id, v.item())] += 1

 

  return samples, cpds_new

# %% [markdown]
# `get_true_prob_joint`

# %%
def get_true_prob_joint(ids, values, prob_dict):
    ''' Create joint distribution from CPDs in prob_dict '''

    id_values = {k:v for k,v in zip(ids,values)}
    top_order = cpds.g.get_topological_order()
    
    top_order_selected_ids = [n.id for n in top_order if n.id in ids]
    top_order_selected_values = [id_values[n.id] for n in top_order if n.id in ids]
    
    answer = 1
    for id_V, v in list(zip(top_order_selected_ids, top_order_selected_values)):
        answer *= get_true_prob(
            id_V=id_V, 
            pa_V=get_parent_values_constrained(id_V=id_V, cA=top_order_selected_values, cB=[], cpds=cpds), 
            prob_dict=prob_dict
        )[v]
    
    return answer

# %% [markdown]
# `evaluate_rewards`

# %%
def evaluate_rewards(N, g, cpds, prob_dict, num_sub_runs):
  ''' Evaluate expected rewards of learned policy after updating beliefs '''


  total_answer = 0

  for _ in range(num_sub_runs): 
    samples, cpds_new = get_samples_and_update_beliefs(N, g, cpds, prob_dict)

    ## For each cA, compute x* (using get E_hat[Y|do(x*), cA])
    opt_actions = {}

    CA_ids = cpds.g.get_list_of_CA_ids()
  
    CA_possible_values = cartesian([cpds.g.get_node_by_id(id_V=_).values for _ in CA_ids])

    for cA in CA_possible_values:
      best_action = np.random.choice(g.get_node_by_id(id_V='X').values) # Pick random x as start candidate, to avoid bias
      best_action_reward = E_hat_do_x_cA(id_V='Y', x=best_action, cA=cA, cpds=cpds_new)

      for x in g.get_node_by_id(id_V='X').values:
        if E_hat_do_x_cA(id_V='Y', x=x, cA=cA, cpds=cpds_new) > best_action_reward:
          best_action = x
          best_action_reward = E_hat_do_x_cA(id_V='Y', x=x, cA=cA, cpds=cpds_new)

      s = [str(_) for _ in cA]
      opt_actions[f"[{' '.join(s)}]"] = int(best_action)


    # Finally compute Total exp rewards = \sum_{c1} E[Y|do(x*), c1] * P(c1)    
    answer = 0

    for cA in CA_possible_values:
      s = [str(_) for _ in cA]
      opt_action = opt_actions[f"[{' '.join(s)}]"]
      answer += E_true_do_x_cA(opt_action, cA, cpds=cpds_new, prob_dict=prob_dict) * get_true_prob_joint(ids=CA_ids, values=cA, prob_dict=prob_dict)

    total_answer += answer


  return opt_actions, total_answer/num_sub_runs





# %% [markdown]
# ## Full experiments (multiple runs)

# %% [markdown]
# ### Set up
# 

# %%
from run_params import B_VALUES, NUM_RUNS_PER_B, NUM_SUB_RUNS  # Import run settings

# %%
# Number of samples in D_L
WARM_START_SAMPLES = int(num_value_c1*num_value_x/2) 

# Verbosity flag
VERBOSE = True

params_for_runs = {
    'B' : sum([[b]*NUM_RUNS_PER_B  for b in B_VALUES], []),
    'max_N' : [30]*NUM_RUNS_PER_B*len(B_VALUES)
}

assert len(params_for_runs['B'])==NUM_RUNS_PER_B*len(B_VALUES)
assert len(params_for_runs['max_N'])==NUM_RUNS_PER_B*len(B_VALUES)

# Some settings for the scipy optimizer; can be left as is
opt_params = {}
opt_params['mutation']=(0.1, 1.5)
opt_params['maxiter']=500


# %%
## Append experiment-level param to experiment tags in mflow
if MLFLOW_ENABLED:
    exp = mlflow.get_experiment_by_name(experiment_name)
    mlflow.set_experiment_tags({
        **exp.tags,
        **params_for_runs,
        **opt_params
    })

# %%
from scipy.optimize import differential_evolution

# %% [markdown]
# ### Run

# %%
CA_ids = cpds.g.get_list_of_CA_ids()
x_and_CA_possible_values = cartesian([cpds.g.get_node_by_id(id_V=_).values for _ in ['X']+CA_ids])

# %% [markdown]
# Our algorithm

# %%
def run_our_algo(B, max_N, prob_dict):
    ''' Run our algorithm '''
    Nfeval = 1

    nlc = NonlinearConstraint(
        con, 
        np.array([0, 1]), 
        np.array([B, np.inf])
    )  # 0 <= sum of beta <= B && 1 <= sum of N <= np.inf
   

    bounds = Bounds(
        [0]*len(x_and_CA_possible_values), 
        [max_N]*len(x_and_CA_possible_values)
    )  # 0 <= N_{x,c} <= np.inf, for each (x,c)


    callback = callbackF if VERBOSE else None

    res_int = differential_evolution(
        func=fun,
        mutation=opt_params['mutation'],
        x0=[np.random.randint(2) for _ in range(len(x_and_CA_possible_values))],
        maxiter=opt_params['maxiter'],
        bounds = bounds,
        vectorized=True,
        constraints=nlc,
        integrality=[True]*len(x_and_CA_possible_values),
        # callback=callback
    )

    if res_int.success == False:  
        res_int.x = [0.000e+00]*len(x_and_CA_possible_values)
    

    opt_result_ourAlgo_INT = {
        "x" : list(res_int.x),
        "fun" : fun(list(res_int.x)),
        "con" : con(list(res_int.x))
    }

    if MLFLOW_ENABLED:
        mlflow.log_dict(opt_result_ourAlgo_INT, "opt_result_ourAlgo_INT.json")

    
    policy, reward = evaluate_rewards(N=get_integer_solutions(res_int.x), g=g, cpds=cpds, prob_dict=prob_dict, num_sub_runs=NUM_SUB_RUNS)

    if MLFLOW_ENABLED:
        mlflow.log_dict(policy, "policy_ourAlgo.json")
        mlflow.log_metric("reward_ourAlgo", reward)


# %% [markdown]
# Baseline 2

# %%
def run_baseline2(B, max_N, prob_dict):
    ''' Run baseline EqualAlloc '''

    Nfeval = 1

    nlc_scalar = NonlinearConstraint(
        con_scalar, 
        np.array([0, 1]), 
        np.array([B, np.inf])
    )  # 0 <= sum of beta <= B && 1 <= sum of N <= np.inf

    bounds_scalar = Bounds([0], [max_N])  # 0 <= N_* <= np.inf


    res_baseline_2 = differential_evolution(
        func=lambda N : 1/np.sum(N, axis=0),
        x0=[np.random.randint(2)],
        mutation=opt_params['mutation'],
        maxiter=opt_params['maxiter'],
        bounds = bounds_scalar,
        constraints=nlc_scalar,
        integrality=[True],
        # callback=callbackF
    )

    if res_baseline_2.success == False:  
        res_baseline_2.x = [0.000e+00]   


    opt_result_Baseline2 = {
        "x" : list(res_baseline_2.x),
        "fun" : fun_scalar(list(res_baseline_2.x)),
        "con" : con_scalar(list(res_baseline_2.x))
    }

    if MLFLOW_ENABLED:
        mlflow.log_dict(opt_result_Baseline2, "opt_result_Baseline2.json")

    policy, reward = evaluate_rewards(N=get_integer_solutions(res_baseline_2.x)*len(x_and_CA_possible_values), g=g, cpds=cpds, prob_dict=prob_dict, num_sub_runs=NUM_SUB_RUNS)


    if MLFLOW_ENABLED:
        mlflow.log_dict(policy, "policy_baseline2.json")
        mlflow.log_metric("reward_baseline2", reward)

# %% [markdown]
# Baseline 3

# %%
def run_baseline3(B, max_N, prob_dict):
    ''' Run baseline MaxSum '''

    Nfeval = 1

    nlc = NonlinearConstraint(
        con, 
        np.array([0, 1]), 
        np.array([B, np.inf])
    )  # 0 <= sum of beta <= B && 1 <= sum of N <= np.inf


    bounds = Bounds(
        [0]*len(x_and_CA_possible_values), 
        [max_N]*len(x_and_CA_possible_values)
    )  # 0 <= N_{x,c} <= np.inf, for each (x,c)

    res_baseline_3 = differential_evolution(
        func=lambda N : 1/np.sum(N, axis=0),
        x0=[np.random.randint(2) for _ in range(len(x_and_CA_possible_values))],
        vectorized=True,
        maxiter=opt_params['maxiter'],
        bounds = bounds,
        mutation=opt_params['mutation'],
        constraints=nlc,
        integrality=[True]*len(x_and_CA_possible_values),
    )



    if res_baseline_3.success == False:  
        res_baseline_3.x = [0.000e+00]*len(x_and_CA_possible_values)

    opt_result_Baseline3 = {
        "x" : list(res_baseline_3.x),
        "fun" : fun(list(res_baseline_3.x)),
        "con" : con(list(res_baseline_3.x))
    }

    if MLFLOW_ENABLED:
        mlflow.log_dict(opt_result_Baseline3, "opt_result_Baseline3.json")

    
    policy, reward = evaluate_rewards(N=get_integer_solutions(res_baseline_3.x), g=g, cpds=cpds, prob_dict=prob_dict, num_sub_runs=NUM_SUB_RUNS)


    if MLFLOW_ENABLED:
        mlflow.log_dict(policy, "policy_baseline3.json")
        mlflow.log_metric("reward_baseline3", reward)


# %% [markdown]
# Baseline 4

# %%
def get_phat_joint(ids, values, cpds):
    ''' A utility function for baseline PropToValue '''

    id_values = {k:v for k,v in zip(ids,values)}
    top_order = cpds.g.get_topological_order()
    
    top_order_selected_ids = [n.id for n in top_order if n.id in ids]
    top_order_selected_values = [id_values[n.id] for n in top_order if n.id in ids]
    
    answer = 1
    for id_V, v in list(zip(top_order_selected_ids, top_order_selected_values)):
        answer *= P_hat(
            id_V=id_V, 
            v=v, 
            pa_V=get_parent_values_constrained(id_V=id_V, cA=top_order_selected_values, cB=[], cpds=cpds), 
            cpds=cpds
        )
    
    return answer

# %%
def con_baseline4_helper(N_scalar):
    ''' Another utility function for baseline PropToValue '''

    CA_ids = cpds.g.get_list_of_CA_ids()     
    x_and_CA_possible_values = cartesian([cpds.g.get_node_by_id(id_V=_).values for _ in ['X']+CA_ids])

    N = np.zeros(len(x_and_CA_possible_values))

    for x_and_cA in x_and_CA_possible_values:
        answer = E_hat_do_x_cA(id_V='Y', x=x_and_cA[0], cA=x_and_cA[1:], cpds=cpds)

        answer *= get_phat_joint(ids=CA_ids, values=x_and_cA[1:], cpds=cpds)

        N[get_N_index(ids=['X']+cpds.g.get_list_of_CA_ids(), values=x_and_cA)] = answer * N_scalar[0]
        
    
    return N


def con_baseline4(N_scalar):
    return con(con_baseline4_helper(N_scalar))

# %%
def run_baseline4(B, max_N, prob_dict):
    ''' Run baseline PropToValue '''

    Nfeval = 1

    nlc = NonlinearConstraint(
        con_baseline4, 
        np.array([0, 1]), 
        np.array([B, np.inf])
    )  # 0 <= sum of beta <= B && 1 <= sum of N <= np.inf

    bounds = Bounds([0], [max_N])  # 0 <= N_scalar <= max_N

    res_baseline_4 = differential_evolution(
        func=lambda N : 1/np.sum(N, axis=0), 
        mutation=opt_params['mutation'],
        x0=[np.random.randint(2)],
        maxiter=opt_params['maxiter'],
        bounds = bounds,
        constraints=nlc,
        integrality=[None],
    )


    if res_baseline_4.success == False:  
        res_baseline_4.x = [0.000e+00]   

    opt_result_Baseline4 = {
        "x" : list(res_baseline_4.x),
        "N" : list(con_baseline4_helper(res_baseline_4.x)),
        "fun" : 1/res_baseline_4.x[0],
        "con" : con_baseline4(list(res_baseline_4.x))
    }

    if MLFLOW_ENABLED:
        mlflow.log_dict(opt_result_Baseline4, "opt_result_Baseline4.json")

    
    policy, reward = evaluate_rewards(
        N=get_integer_solutions(con_baseline4_helper(res_baseline_4.x)), 
        g=g, 
        cpds=cpds, 
        prob_dict=prob_dict, 
        num_sub_runs=NUM_SUB_RUNS
    )


    if MLFLOW_ENABLED:
        mlflow.log_dict(policy, "policy_baseline4.json")
        mlflow.log_metric("reward_baseline4", reward)


# %% [markdown]
# Call all algorithms

# %%
def reset_beliefs():
    ''' Reset beliefs for each CPD '''

    global cpds
    cpds = CPDCollection(g)

# %%
print(f"MLFLOW_ENABLED = {MLFLOW_ENABLED}\n")

# %%
for i in range(NUM_RUNS_PER_B*len(B_VALUES)):
    print(f"\nRun number: {i}")

    # global prob_dict
    prob_dict = sample_true_prob()

    mlflow.log_dict(prob_dict, "prob_dict.json")

    seed = np.random.randint(10000)
    np.random.seed(seed)

    if MLFLOW_ENABLED:
        v = mlflow.log_param("numpy seed", seed)
        assert v == seed
        
        mlflow.log_artifact("env_setting_4.py")

        mlflow.log_param("B", params_for_runs['B'][i])
        mlflow.log_param("max_N", params_for_runs['max_N'][i])

        mlflow.log_param("NUM_RUNS_PER_B", NUM_RUNS_PER_B)
        mlflow.log_param("NUM_SUB_RUNS", NUM_SUB_RUNS)
        mlflow.log_param("WARM_START_SAMPLES", WARM_START_SAMPLES)

        tags = {
            'num_value_c0' : num_value_c0
        }
        mlflow.set_tags(tags)
    
    print(f"Sampling D_L...")
    reset_beliefs()
    warm_start(num_init_samples=WARM_START_SAMPLES, prob_dict=prob_dict)

    print(f"Running our algorithm...")
    run_our_algo(params_for_runs['B'][i], params_for_runs['max_N'][i], prob_dict)
    
    print(f"Running baseline EqualAlloc...")
    run_baseline2(params_for_runs['B'][i], params_for_runs['max_N'][i], prob_dict)

    print(f"Running baseline MaxSum...")
    run_baseline3(params_for_runs['B'][i], params_for_runs['max_N'][i], prob_dict)

    print(f"Running baseline PropToValue...")
    run_baseline4(params_for_runs['B'][i], params_for_runs['max_N'][i], prob_dict)

    if MLFLOW_ENABLED:
        mlflow.end_run()


