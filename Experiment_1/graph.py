import numpy as np
from misc_utils import cartesian


######################################################################

class Node:
  def __init__(self, id_V, PA_V, values):
    self.id = id_V
    # self.parent_ids = id_PA_V
    self.values = values
    self.parents = PA_V
    self.children = []

  def copy(self):
    n_copy = Node(id_V=self.id, PA_V=self.parents.copy(), values=self.values.copy())
    n_copy.children = self.children.copy()
    return n_copy
  
  def __str__(self):
    return f"{self.parents} --> {self.id} : {self.values}"


######################################################################

class Graph:
  def __init__(self):
    self.nodes = []
    self.roots = []
    self.CA = []
    self.CB = []

  def get_node_by_id(self, id_V):
    for n in self.nodes:
      if n.id == id_V:
        return n
    
    return None

  def get_list_of_CA_ids(self):
    return [_.id for _ in self.CA]

  def get_list_of_CB_ids(self):
    return [_.id for _ in self.CB]


  def insert(self, V, PA_V, values, cat=None):
    PA = []
    for p in PA_V:
      PA += [self.get_node_by_id(p)]

    n = Node(V, PA, values)

    self.nodes += [n]

    if len(PA_V) == 0:
      self.roots += [n]

    if cat=='CA':
      self.CA += [n]
    elif cat=='CB':
      self.CB += [n]

    for p in PA:
      p.children += [n]

  def get_RVs(self):
    rvs = []
    for n in self.nodes:
      rvs += [n.id]

    return rvs

  def get_roots(self):
    return self.roots

  def remove_edge(self, id_a, id_b):
    '''Remove edge a->b'''
    a = self.get_node_by_id(id_a)
    b = self.get_node_by_id(id_b)
    a.children.remove(b)
    b.parents.remove(a)

    return


  def copy(self):
    g_copy = Graph()
    for n in self.nodes:
        if n.id in self.get_list_of_CA_ids():
            cat = 'CA'
        elif n.id in self.get_list_of_CB_ids():
            cat = 'CB'
        else:
            cat = None
        g_copy.insert(n.id, [_.id for _ in n.parents], n.values.copy(), cat=cat)

    return g_copy

  def get_topological_order(self):
    g_copy = self.copy()

    S = g_copy.roots
    L = []

    while len(S)>0:
        n = S.pop()
        L += [n]

        for m in n.children.copy():
            n.children.remove(m)
            m.parents.remove(n)

            if len(m.parents)==0:
                S += [m]
    return L



  def __str__(self):
    output = f""
    for n in self.nodes:
      parent_ids = [pa.id for pa in n.parents]
      output += f"{parent_ids} --> {n.id} : {n.values}\n"

    return output



######################################################################




class CPD:
  def __init__(self, g, id_V, id_PA_V, values_pa_V):
    self.g = g # Graph
    self.id_V = id_V
    self.id_PA_V = id_PA_V
    self.values_pa_V = values_pa_V
    self.beliefs = [1]*len(g.get_node_by_id(id_V).values)

  def __str__(self):
    return f"id_V = {self.id_V}\n" + f"id_PA_V = {self.id_PA_V}\n" + f"values_pa_V = {self.values_pa_V}\n" + f"beliefs = {self.beliefs}"

  def copy(self):
    cpd_new = CPD(self.g, self.id_V, self.id_PA_V, self.values_pa_V)
    cpd_new.beliefs = self.beliefs.copy()
    return cpd_new


######################################################################


class CPDCollection:
  def __init__(self, g):
    self.g = g
    self.cpds = {}

    for node in g.nodes:
      if len(node.parents) != 0:
        possible_pa_values = cartesian([p.values for p in node.parents])
        parent_ids = [pa.id for pa in node.parents]

        for pa_value in possible_pa_values:
          self.cpds[f"{node.id}_{pa_value}"] = CPD(g=g, id_V=node.id, id_PA_V=parent_ids, values_pa_V=pa_value)
      else:
        self.cpds[f"{node.id}_[]"] = CPD(g=g, id_V=node.id, id_PA_V=[], values_pa_V=np.array([]))


  def get_cpd(self, id_V, values_pa_V):
    s = [str(_) for _ in values_pa_V]
    return self.cpds[f"{id_V}_[{' '.join(s)}]"]


  def copy(self):
    cpdcollection_new = CPDCollection(self.g)
    cpdcollection_new.cpds = {}
    for k, v in self.cpds.items():
      cpdcollection_new.cpds[k] = v.copy()
    return cpdcollection_new


      


