import numpy as np
import networkx as nx
import scipy.sparse as spar
import pickle
from scipy import linalg as la

def cosine_similarity(u, v):
    """ Calculates the cosine similarity of two vectors u and v.
    Input:
        u (array) : first vector 
        v (array) : second vector
    Returns:
        t (float) : angle between the vectors
    """
    t = u.dot(v)/(la.norm(u)*la.norm(v))
    return t

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return spar.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def directed_graph_pickle(filename, G):
    pickle.dump(G, open(filename, 'w'))

def load_directed_graph_pickle(filename):
    return pickle.load(open(filename))

if __name__ == "__main__":
    print "This is a libary, not an executable script"
