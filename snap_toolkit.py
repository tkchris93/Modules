import numpy as np
import networkx as nx
import scipy.sparse as spar

def edges_to_sparse_matrix(filename):
    data = np.loadtxt(filename)

    adjdict = dict()

    data = data.astype(int)
    for i in xrange(data.shape[0]):
        if adjdict.has_key(data[i,0]):
            adjdict[data[i,0]].append(data[i,1])
        else:
            adjdict[data[i,0]] = [data[i,1]]
    for i in xrange(data.shape[0]):
        if not adjdict.has_key(data[i,1]):
            adjdict[data[i,1]] = []

    nodes_set = set()
    for i in xrange(data.shape[0]):
        for j in xrange(data.shape[1]):
            nodes_set.add(data[i,j])
    nodes_list = list(nodes_set)
    values = dict()
    for i in xrange(len(nodes_list)):
        values[nodes_list[i]] = i

    refactored = dict()

    for node in adjdict.keys():
        edges = adjdict[node]
        refactored_edges = []
        for e in edges:
            refactored_edges.append(values[e])
        refactored[values[node]] = refactored_edges

    G = nx.from_dict_of_lists(refactored, create_using=nx.DiGraph())

    n = len(G.nodes())
    A = spar.lil_matrix((n,n))
    for u in G.nodes():
        for v in refactored[u]:
            A[u,v] = 1

    return A
