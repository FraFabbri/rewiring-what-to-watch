

import numpy as np
import glob
import pickle
import random

from absorbing import compute_radicalization

from tqdm import tqdm
from scipy.sparse import csr_matrix




### function to creates synthetic recommendations 
def create_synthetic_recommendations(graph):
    
    def find_bridges(graph, t):
        
        u,v = t[0], t[1]
        
        outlinks_from_u = set(graph.neighborhood(vertices=u, mode="out"))
        
        inlinks_to_v = set(graph.neighborhood(vertices=v, mode="in"))
        
        bridges = outlinks_from_u.intersection(inlinks_to_v)
        
        return list(bridges)
    
    
    
    def find_nodes_at_distance_2(graph):

        dict_nodes_at_dist2 = {}

        for n in graph.vs:

            nodes_at_distance_2 = graph.neighborhood(vertices=n, order=2, mode="out", mindist = 2)

            if nodes_at_distance_2 != []:

                dict_nodes_at_dist2[n.index] = nodes_at_distance_2

        return dict_nodes_at_dist2
    
    
    def compute_DirectedAdamicAdar(graph, t):
        
        bridges = find_bridges(graph, t)
        
        out = list(map(lambda n: 1./np.log2(len(graph.neighborhood(vertices=n, mode="out"))+1), bridges))
        
        return sum(out)    
    
    
    dict_nodes_at_dist2 = find_nodes_at_distance_2(graph)
    
    mapping_scores = {}
    
    
    for u in dict_nodes_at_dist2:
        
        mapping_scores[u] = []
        
        for v in dict_nodes_at_dist2[u]:
            
            ada_ = compute_DirectedAdamicAdar(graph, (u,v))
            
            mapping_scores[u].append((u, v, ada_))
        
    
        mapping_scores[u] = sorted(mapping_scores[u], key=lambda x: x[1], reverse=True)
    
    return mapping_scores



### class to run algorithms to reduce radicalization 

class AbsorbingRandomWalk():
    
    def __init__(self, graph, abs_label, d_outdegree, mapping_relevance_scores, fn_f_matrix):
        
        self._graph = graph
        
        self._absorbing_label = abs_label
        
        self._d_outdegree = d_outdegree
        
        
        # fix tau to 0 to avoid using relevance constraint
        self._tau = 0
        
        self._mapping_relevance_scores = mapping_relevance_scores
                
        # fn F_matrix
        self._fn_f_matrix = fn_f_matrix
        
        # take absorbing and transient idxs
        self._absorbing_nodes = set([n.index for n in self._graph.vs if n["label"] != self._absorbing_label])
        self._transient_nodes = [n.index for n in self._graph.vs if n["label"] == self._absorbing_label]
        
        # useful mapping for compute F and z
        self._mapping_transient_to_matrix = {idx: position for position, idx in enumerate(self._transient_nodes)}
        self._mapping_matrix_to_transient = {position: idx for position, idx in enumerate(self._transient_nodes)}
        
        
        # mapping probabilities
        
        self._prob_edges = {}
        
        for e in self._graph.es:
            
            _source = e.source
            _target = e.target
            _pr = e["weight"]
            
            if _source not in self._prob_edges:
                
                self._prob_edges[_source] = {"target": {}, "tot": 0}
                
            
            self._prob_edges[_source]["target"][_target] = _pr
            self._prob_edges[_source]["tot"] +=1
            
        for s in self._prob_edges:
            
            tot = self._prob_edges[_source]["tot"]
            
            for t in self._prob_edges[s]["target"]:
                
                self._prob_edges[s]["target"][t] = self._prob_edges[s]["target"][t]/tot
        
        

    
    def initialize(self, damping_factor=False):
                
        # initialization before starting the rewiring
        
        if glob.glob(self._fn_f_matrix) == []:
            
            print("creating-and-saving-F")
            
            self._F_matrix, self._z_vec = compute_radicalization(self._graph, self._transient_nodes, damping_factor)

            with open(self._fn_f_matrix, "wb") as f:
                
                sparse_F_matrix = csr_matrix(self._F_matrix)
                
                pickle.dump(sparse_F_matrix, f)

        else:
            
            with open(self._fn_f_matrix, "rb") as f:

                self._F_matrix = pickle.load(f) 
                
                # convert sparse matrix to full
                self._F_matrix = self._F_matrix.todense()
                
                # initialize the z-vec                               
                self._z_vec = self._F_matrix.sum(1)
        
        # initialize the maximum
        self._z_max = self._z_vec.max()
        
        # needed to store all the potential rewirings
        self._R_potential_solutions = set()

        # needed for tracking values of OurAlgorithm with different K
        self._history_optimal_r = []
        self._history_z_max = []
        
        # needed for the two *Baselines*
        self._1st_all_deltas = {}
        self._1st_all_sorted_deltas = []
        self._1st_z_vec = self._z_vec.copy()
            
        self._1st_F_matrix = csr_matrix(self._F_matrix)
        
        

        
        
    def compute_delta_vec(self, t_uvw):

        idx_u, idx_v, _  = t_uvw

        position_u = self._mapping_transient_to_matrix[idx_u]

        position_v = self._mapping_transient_to_matrix[idx_v]

        f_uv = self._F_matrix[position_u, position_v] 


        z_v = self._z_vec[position_v]

        f_u = self._F_matrix[:, position_u]

        p_r = self._prob_edges[idx_u]["target"][idx_v]

        delta_vec = (f_u*z_v)/(1/p_r + f_uv)

        return delta_vec
        
 
    def find_the_candidates_1rewiring(self):

        # INITIALIZATION # 
        
        if self._R_potential_solutions == set():

            R_potential_solutions = set()

            for u_node in self._transient_nodes:

                name_u_node = self._mapping_id_names[u_node]

                left_scores = self._mapping_relevance_scores[name_u_node]#[self._d_outdegree:]

                potential_uw = None

                # 0. select all the potential (u,w) tuples

                for _, name_w_node, s_uw in left_scores:

                    if name_w_node in self._mapping_names_id:

                        w_node = self._mapping_names_id[name_w_node]

                        if w_node in self._absorbing_nodes:

                            potential_uw = (w_node, s_uw)

                            break

                if potential_uw != None:

                    selected_w, selected_s_uw = potential_uw
                    
                    # to improve with dictionary
                    single_edgelist = [(e.target, e["weight"]) for e in self._graph.es.select(_source=u_node)]


                    # select the potential (u,v,w) filtering from (u,v) tuples
                    
                    for v_node, s_uv in single_edgelist:


                        if v_node in self._transient_nodes:

                            # fix tau to np.inf to avoid using relevance constraint
                            if selected_s_uw/s_uv >= self._tau and u_node != v_node:

                                R_potential_solutions.update([(u_node, v_node, selected_w)])            

            self._R_potential_solutions = R_potential_solutions
            
            self._1st_R_potential_solutions = R_potential_solutions
            
            
        
    def find_the_optimal_1rewiring(self):
        
        # 2. find the optimal rewiring

        delta_max = -np.inf

        optimal_rewiring = None
                            
        for r in tqdm(self._R_potential_solutions):


            # find the top-rewiring

            delta_vec = self.compute_delta_vec(r)

            z_1 = self._z_vec - delta_vec
            
            #print(delta_vec.shape)
            
            new_max = z_1.max()

            one_delta = self._z_max - new_max
            
            if r not in self._1st_all_deltas:
            
                self._1st_all_deltas[r] = (one_delta, 
                                           
                                           #delta_vec
                                          )

            # check the optimal delta

            if delta_max < one_delta:

                delta_max = one_delta

                optimal_rewiring = r
        
        
        self._delta_max = delta_max
        
        self._optimal_r = optimal_rewiring
                
        self._R_potential_solutions.remove(optimal_rewiring)
        
        self._history_optimal_r.append(self._optimal_r)       
        

    def update_solution(self, F_zvec_zmax):
        
        self._F_matrix = []
        
        self._F_matrix = F_zvec_zmax[0].todense()
        
        #print(self._F_matrix.sum())
        
        self._z_vec = F_zvec_zmax[1]
        
        self._z_max = F_zvec_zmax[2]
        
        self._history_z_max.append(self._z_max)

        ###
        

    def apply_1rewiring(self, optimal_r=None, F_matrix=[]):
        

        # initially, but now NO -> (modify the graph)
        
        # check when we update the original one

        if optimal_r == None:
            
            print("Use the optimum")
            
            optimal_r = self._optimal_r
            
            print(optimal_r)
            
        # check when we update the original one
        if F_matrix == []:
        
            F_matrix = csr_matrix(self._F_matrix)

            
        idx_u, idx_v, idx_w  = optimal_r
            
            
        old_weight = self._graph.es.select(_source=idx_u, _target = idx_v)["weight"][0]

        #self._graph.delete_edges([(idx_u,idx_v)])
        #self._graph.add_edge(source=idx_u, target = idx_w, weight = old_weight)
        

        # position of u,v,w in the matrix F
        position_u = self._mapping_transient_to_matrix[idx_u]
        position_v = self._mapping_transient_to_matrix[idx_v]

        e_array = np.zeros(len(self._transient_nodes), dtype=np.float32)
        e_array[position_u] = +1

        g_array = np.zeros(len(self._transient_nodes), dtype=np.float32)
        g_array[position_v] = +1.*(old_weight)
    
        e_array = csr_matrix(e_array.reshape(1, len(e_array)).T)
        g_array = csr_matrix(g_array.reshape(1, len(g_array)).T)

        matrix_eg = csr_matrix(np.dot(e_array, g_array.T))

        update_F = np.dot(F_matrix, matrix_eg)

        update_F = np.dot(update_F, F_matrix)
        
        den = 1./old_weight + F_matrix[position_v, position_u] 

        update_F = update_F/den
        
        F_matrix = F_matrix - update_F
        
        
        return F_matrix, F_matrix.sum(1), F_matrix.sum(1).max(), 



        

    def compute_AllInOneByDelta(self, K):
        """
        We select K rewirings coming from OneRewiring Algorithm
        """

        # find the K

        if self._1st_all_sorted_deltas == []:
            
            if self._1st_all_deltas == {}:
                
                # update delta vectors and delta values
                
                print("recover delta vectors and values")
                
                for r in tqdm(self._R_potential_solutions):


                    # find the top-rewiring

                    delta_vec = self.compute_delta_vec(r)

                    z_1 = self._z_vec - delta_vec

                    new_max = z_1.max()

                    one_delta = self._z_max - new_max

                    if r not in self._1st_all_deltas:

                        self._1st_all_deltas[r] = (one_delta, 
                                                   #delta_vec
                                                  )

        

            # order delta-value, delta-vector
            self._1st_all_sorted_deltas = sorted([(r, self._1st_all_deltas[r]) 

                                                  for r in self._1st_all_deltas], key = lambda v: v[1][0], reverse=True)



        # select the top-K rewirings
        top_k_rewirings = self._1st_all_sorted_deltas[:K]

        # load the first F_matrix
        F_matrix = csr_matrix(self._1st_F_matrix)
        
        
        history_z = [self._1st_z_vec]
        
        for (r, (d, )) in top_k_rewirings:

            F_matrix__z_vec__z_max = self.apply_1rewiring(optimal_r=r, F_matrix=F_matrix)
            

            F_matrix, z_vec, z_max = F_matrix__z_vec__z_max
            
            history_z.append(z_vec)
            
            del F_matrix__z_vec__z_max


        return history_z
        
        
    def compute_AllInOneByZ(self, K, F_matrix):
        """
        We select rewirings coming from only top-K nodes in z 
        """

        
        if self._1st_all_sorted_deltas == []:
            
            if self._1st_all_deltas == {}:
                
                # update delta vectors and delta values
                
                print("recover delta vectors and values")
                
                for r in tqdm(self._R_potential_solutions):


                    # find the top-rewiring

                    delta_vec = self.compute_delta_vec(r)

                    z_1 = self._z_vec - delta_vec

                    new_max = z_1.max()

                    one_delta = self._z_max - new_max

                    if r not in self._1st_all_deltas:

                        self._1st_all_deltas[r] = (one_delta, 
                                                   #delta_vec
                                                  )

                

            # order delta-value, delta-vector
            self._1st_all_sorted_deltas = sorted([(r, self._1st_all_deltas[r]) 

                                                  for r in self._1st_all_deltas], key = lambda v: v[1][0], reverse=True)

        
        # we select for each of the K nodes (ordered by z-value) the best possible rewiring

        top_k_nodes_by_position = sorted(enumerate(self._1st_z_vec), key = lambda x: x[1], reverse=True)[:K]


        top_k_nodes_by_id = set([self._mapping_matrix_to_transient[position] for position, value in top_k_nodes_by_position])


        top_deltas_by_k_nodes = [r for (r, (one_delta, 
                                            #delta_vec
                                           )) in self._1st_all_sorted_deltas 
                                 if r[0] in top_k_nodes_by_id 
                                ][:K]

        # initialization
        checked_nodes = top_k_nodes_by_id.copy()
        
        z_vec = None
        
        
        for r in top_deltas_by_k_nodes:
            
            F_matrix__z_vec__z_max = self.apply_1rewiring(optimal_r=r, F_matrix=F_matrix)

            F_matrix, z_vec, z_max = F_matrix__z_vec__z_max

            del F_matrix__z_vec__z_max

        return z_vec         
        
        
        
        
        
        
    def compute_random(self, K, F_matrix):
        
        """
        K random rewirings, constrained by the nDCG metric, we sample edges from B-to-B and generate a new edge sample at random from B-to-G
        """
        
        
        selected_rewirings = random.sample(population=list(self._1st_R_potential_solutions), k=K)
        
        history_z = [self._1st_z_vec]
        
        F_matrix = csr_matrix(self._1st_F_matrix)

        
        for r in selected_rewirings:
            
            F_matrix__z_vec__z_max = self.apply_1rewiring(optimal_r=r, F_matrix=F_matrix)

            F_matrix, z_vec, z_max = F_matrix__z_vec__z_max
            
            del F_matrix__z_vec__z_max
            
            history_z.append(z_vec)

                

        return history_z

    
    def compute_RL(self, K_max, sorted_centralities):
        """
        K rewiring from K nodes having the highest RepBubLink score
        """

        selected_nodes_by_RL = [n for (n,s) in sorted_centralities][:K_max]

        potential_tuples = []

        for n in selected_nodes_by_RL[:K_max]:

            potential_tuples += [t for t in self._R_potential_solutions if t[0] == n]

            if len(potential_tuples) >= K_max:

                break


        selected_rewirings = potential_tuples[:K_max]

        F_matrix = csr_matrix(self._1st_F_matrix)


        if selected_rewirings != []:

            for r in selected_rewirings:

                F_matrix__z_vec__z_max = self.apply_1rewiring(optimal_r=r, F_matrix=F_matrix)

                F_matrix, z_vec, z_max = F_matrix__z_vec__z_max

                del F_matrix__z_vec__z_max


            return z_vec

        else:

            return self._1st_z_vec