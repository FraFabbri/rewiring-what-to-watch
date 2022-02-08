


#from scipy.sparse import eye
#from scipy.sparse.linalg import spilu

from sklearn.preprocessing import normalize
from myutils import get_sparse_adj_martrix

from tqdm import tqdm 


def approximate_inverse(MM_sparse, step=200):
    
    current_ = MM_sparse.copy()

    tot_ = MM_sparse.copy()

    z_old = tot_.sum(1).max()    
    
    # initialize counter

    
    iterations = 5000
    
    for counter in tqdm(range(iterations)):

        #print(_)
        current_ = current_.dot(MM_sparse)

        tot_ = tot_ + current_

        z_max = tot_.sum(1).max()


        if counter%step == 0:

            ratio = z_old/z_max
            
            print(z_max)
            print(ratio)

            print("---")
            z_old = z_max    
            
            
            if ratio >= 0.98:
                
                break
                
                
    return tot_


def return_adj_matrix(graph):
    
    """
    from graph to adj-matrix
    """
    
    edgelist_idx = graph.get_edgelist()
    
    if "weight" not in graph.es.attributes():
        
        weights = [1]*len(edgelist_idx)
        
    else:
        
        weights = graph.es["weight"]
        
    N = graph.vcount()

    adj_matrix = get_sparse_adj_martrix(edgelist_idx, weights, N)
    
    #return adj_matrix
    adj_matrix = normalize(adj_matrix, norm='l1', axis=1)
    
    return adj_matrix




def compute_radicalization(graph, transient_nodes, damping_factor=False, sparse=False):    
    
    # [1] - transient nodes
    adj_matrix = return_adj_matrix(graph)
    
    # [2] - A_tt matrix
    A_tt = adj_matrix[transient_nodes, :][:, transient_nodes]#[np.ix_(transient_nodes,transient_nodes)]

    # [3] - Damping vector
    if damping_factor:
        
        val_damping = graph.vcount()/graph.vcount()

        #damping_vector = damping_vector[transient_nodes]

        #damping_vector[~np.isfinite( damping_vector )] = 0
        #damping_vector = (1-damping_factor)*damping_vector

        # [4bis] - Damping factor and sparse transformation
        #damping_vector = csr_matrix(damping_vector.reshape(1,len(damping_vector)))

        A_tt = damping_factor*A_tt + (1-damping_factor)*val_damping


    # [4] compute inverse-matrix  = (I - A_tt)^-1
    
    A_tt = A_tt*-1.

    A_tt = A_tt.todense()

    for ix in range(A_tt.shape[0]):

        A_tt[ix, ix] +=1.

    F = A_tt.getI()
    
    z_vector = F.sum(1)
    
    # [5] generate final-vector for transient-nodes         
    #z_vector = np.array([x[0] for x in z_vector.tolist()])    
    
    return F, z_vector


