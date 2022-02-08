from scipy.sparse import csr_matrix

import numpy as np
import pickle

def get_sparse_adj_martrix(edgelist, weights, N):
    adjacency_matrix = csr_matrix((weights, zip(*edgelist)), shape = (N,N), dtype=np.float32)
    return adjacency_matrix




def save_results(fn_algo, out):
    

    with open(fn_algo, "wb") as f:

        pickle.dump(out, f)
