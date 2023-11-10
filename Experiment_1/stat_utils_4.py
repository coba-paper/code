import numpy as np
# from true_distribution import get_true_prob
from misc_utils import cartesian
from env_setting_4 import num_value_c1, num_value_c0, num_value_x, domain_y


######################################################################

def get_distrib_params(beliefs):
  '''`beliefs` has to be for one specific CPD, not a set of CPDs'''
  total = sum(beliefs)
  return [b/total  for b in beliefs]

######################################################################

def ent(beliefs):
  '''Calculate `Ent`'''
  params = get_distrib_params(beliefs)
  return np.sum([-p*np.log(p) for p in params])

######################################################################

def ent_new(beliefs):
  '''Calculate `Ent-new`'''
  ents_list = []
  for i in range(len(beliefs)):
    ents_list += [ent([beliefs[j]+1 if j==i else beliefs[j] for j in range(len(beliefs))])]

  return np.mean(ents_list)

######################################################################

def get_V_index(V, v):

  if V=='Y':
      return domain_y.index(v)
  else:
      return v

######################################################################

def P_hat(id_V, v, pa_V, cpds):
  ''' Return P(id_V=v | pa_V)'''

  s = [str(_) for _ in pa_V]
  beliefs = cpds.cpds[f"{id_V}_[{' '.join(s)}]"].beliefs

  return beliefs[get_V_index(id_V, v)] / np.sum(beliefs)

######################################################################

def get_parent_values_constrained(id_V, cA, cB, cpds, CA_ids=None, CB_ids=None):
  '''
    If CA_ids and CB_ids are none, the full CA+CB is used.
    Else the given ids are used.
  '''
  n = cpds.g.get_node_by_id(id_V)
  parent_ids = [pa.id for pa in n.parents]
  
  if CA_ids is None and CB_ids is None:
    CA_ids = cpds.g.get_list_of_CA_ids()
    CB_ids = cpds.g.get_list_of_CB_ids()

  idx_in_cA = []
  for p in parent_ids:
      if p in CA_ids:
          idx_in_cA += [CA_ids.index(p)]    
  

  idx_in_cB = []
  for p in parent_ids:
      if p in CB_ids:
          idx_in_cB += [CB_ids.index(p)]    
  

  return list(np.array(cA)[idx_in_cA]) + list(np.array(cB)[idx_in_cB])
    


def E_hat_do_x_cA(id_V, x, cA, cpds):
  ''' E[id_V | do(x), cA]'''

  possible_values = [cpds.g.get_node_by_id(id_V=_).values for _ in cpds.g.get_list_of_CB_ids()]

  answer = 0

  for cB in cartesian(possible_values):
      for y_idx, y in enumerate(cpds.g.get_node_by_id(id_V='Y').values):
          prod_term = 1
          for i in range(len(cB)): # For each element within cB
              id_V = cpds.g.get_list_of_CB_ids()[i]
              values_pa_V = get_parent_values_constrained(id_V=id_V, cA=cA, cB=cB, cpds=cpds)
              prod_term *= get_distrib_params(cpds.get_cpd(id_V=id_V, values_pa_V=values_pa_V).beliefs)[cB[i]]

          id_V = 'Y'
          values_pa_V = get_parent_values_constrained(id_V=id_V, cA=cA, cB=cB, cpds=cpds)
          prod_term *= get_distrib_params(cpds.get_cpd(id_V=id_V, values_pa_V=[x]+values_pa_V).beliefs)[y_idx]

          answer += y * prod_term

  return answer

######################################################################

def get_true_prob(id_V, pa_V, prob_dict):
    s = [str(_) for _ in pa_V]
    return prob_dict[f"{id_V}_[{' '.join(s)}]"]

def E_true_do_x_cA(x, cA, cpds, prob_dict):
  '''Return E[Y|do(x),cA]'''

  possible_values = [cpds.g.get_node_by_id(id_V=_).values for _ in cpds.g.get_list_of_CB_ids()]

  answer = 0

  for cB in cartesian(possible_values):
      for y in cpds.g.get_node_by_id(id_V='Y').values:
          prod_term = 1
          for i in range(len(cB)): # For each element within cB
              id_V = cpds.g.get_list_of_CB_ids()[i]
              values_pa_V = get_parent_values_constrained(id_V=id_V, cA=cA, cB=cB, cpds=cpds)
              prod_term *= get_true_prob(id_V=id_V, pa_V=values_pa_V, prob_dict=prob_dict)[cB[i]]

          id_V = 'Y'
          values_pa_V = get_parent_values_constrained(id_V=id_V, cA=cA, cB=cB, cpds=cpds)
          
          prod_term *= get_true_prob(id_V=id_V, pa_V=[x]+values_pa_V, prob_dict=prob_dict)[get_V_index(id_V, y)]

          answer += y * prod_term

  return answer