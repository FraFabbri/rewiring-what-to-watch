
import igraph
import os
import pickle
import sys
import gc


import pandas as pd

from tqdm import tqdm
from algorithms import *
from config import *

from myutils import save_results

PATH = "../data/youtube/"
label = "bad"


with open("../data/youtube/final/scores-by-d-and-t.p", "rb") as f:

    big_mapping_scores = pickle.load(f)


# options 0,1,2
t_ = int(sys.argv[1])

d_ = int(sys.argv[2])

tau = float(sys.argv[3])

ALGONAMES = sys.argv[4].split("::")


mapping_scores = big_mapping_scores[d_][t_].copy()

del big_mapping_scores


fn_f_matrix = PATH + "inverse/" + "out-%s_t-%s.p"%(d_, str(round(t_/1000)) + "k")

fn_graph = PATH + "graph/" + "graph-%s_t-%s.p"%(d_, str(round(t_/1000)) + "k")



with open(fn_graph, "rb") as f:

    yt_graph = pickle.load(f)

    print(yt_graph.summary())

    

one_instance = AbsorbingRandomWalk(yt_graph, label, d_, mapping_scores, fn_f_matrix)

one_instance._tau = tau

mapping_id_names = {n.index : n["name"] for n in yt_graph.vs}
one_instance._mapping_id_names =  mapping_id_names

mapping_names_id = {n["name"]: n.index for n in yt_graph.vs}
one_instance._mapping_names_id =  mapping_names_id

t_str = str(round(t_/1000)) + "k"



PATH_OUT = "../out/youtube/d%s_t%s_tau%s/"%(d_, t_str, tau)

one_instance.initialize()

print("Selected tau")
print(one_instance._tau)

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)


print("Our Heuristic")
algoname1 = "HEU"
out_fn = PATH_OUT + "algo-%s_maxK-%s.p"
fn_algo1 = out_fn%(algoname1, max_K) 



if algoname1 in ALGONAMES:

    if fn_algo1 not in glob.glob(PATH_OUT+"*"):

        # initialize set of candidates
        print("find-the-candidates")
        one_instance.find_the_candidates_1rewiring()


        # run algorithm1 - our algorithm
        history_z_algo1 = [one_instance._1st_z_vec]

        for one_k in range(max_K):

            print("iteration: %s"%one_k)

            print("find-the-optimal1")
            one_instance.find_the_optimal_1rewiring()

            print("apply-the-rewiring")
            F_matrix__z_vec__z_max  = one_instance.apply_1rewiring()

            print("update-the-solution")
            one_instance.update_solution(F_matrix__z_vec__z_max)

            history_z_algo1.append(F_matrix__z_vec__z_max[1])

            print(F_matrix__z_vec__z_max[-1])

            del F_matrix__z_vec__z_max

        # save results
        save_results(fn_algo1, history_z_algo1)

        del history_z_algo1

    
# run baseline1


print("1st Baseline")
algoname2 = "ByDelta"
fn_algo2 = out_fn%(algoname2, max_K) 


if algoname2 in ALGONAMES:
    
    if fn_algo2 not in glob.glob(PATH_OUT+"*"):

        # initialize set of candidates
        print("find-the-candidates")
        one_instance.find_the_candidates_1rewiring()


        history_z_algo2 = one_instance.compute_AllInOneByDelta(max_K)

        # save results
        save_results(fn_algo2, history_z_algo2)

        del history_z_algo2



print("2nd Baseline")
algoname3 = "ByZ"
fn_algo3 = out_fn%(algoname3, max_K) 


if algoname3 in ALGONAMES:

    if fn_algo3 not in glob.glob(PATH_OUT+"*"):
    #if True:

        # initialize set of candidates
        print("find-the-candidates")
        one_instance.find_the_candidates_1rewiring()


        # run baseline2
        history_z_algo3 = [one_instance._1st_z_vec]

        F_matrix = csr_matrix(one_instance._1st_F_matrix)

        for k in range(1, max_K+1):

            one_vec = one_instance.compute_AllInOneByZ(k, F_matrix)

            history_z_algo3.append(one_vec)

        # save results
        save_results(fn_algo3, history_z_algo3)

        del F_matrix

        del history_z_algo3



print("3rd Baseline")
algoname4 = "Random"
fn_algo4 = out_fn%(algoname4, max_K) 


if algoname4 in ALGONAMES:

    if fn_algo4 not in glob.glob(PATH_OUT+"*"):
    #if True:

        # initialize set of candidates
        print("find-the-candidates")
        one_instance.find_the_candidates_1rewiring()


        #run baseline3
        history_z_algo4 = [one_instance._1st_z_vec]

        F_matrix = csr_matrix(one_instance._1st_F_matrix)


        history_z_algo4 = one_instance.compute_random(max_K, F_matrix)                

        # save results
        save_results(fn_algo4, history_z_algo4)

        del F_matrix

        del history_z_algo4

# clean memory
del one_instance

gc.collect()
