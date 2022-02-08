import igraph
import pandas as pd
import pickle
import glob

from tqdm import tqdm
from algorithms import *

PATH = "../data/youtube/"
label = "bad"



# load all the edgelists and the scores
with open(PATH + "final/edgelist-by-d-and-t.p", "rb") as f:

    edgelist_by_threshold = pickle.load(f)

with open(PATH + "final/scores-by-d-and-t.p", "rb") as f:

    big_mapping_scores = pickle.load(f)

    
nodes_df = pd.read_csv(PATH + "final/yt-nodes.tsv", sep="\t")
nodes_df["category"] = nodes_df["category"].apply(lambda x: x.lower())
mapping_labels = dict(nodes_df[["id", "category"]].values)

## compute the inverse 

lst_thresholds = [5000, 10000, 100000]
lst_outdegree = [5, 10, 20]


for t_ in lst_thresholds:
    
    for d_ in lst_outdegree:

        fn_f_matrix = PATH + "inverse/" + "out-%s_t-%s.p"%(d_, str(round(t_/1000)) + "k")
        
        fn_graph = PATH + "graph/" + "graph-%s_t-%s.p"%(d_, str(round(t_/1000)) + "k")


        print(t_, d_)
        print(fn_f_matrix)

        mapping_scores = big_mapping_scores[d_][t_]        
        
        if glob.glob(fn_graph) == []:
        

            edgelist_ = edgelist_by_threshold[d_][t_]


            yt_graph = igraph.Graph.TupleList(edgelist_, directed=True, edge_attrs="weight")

            for n in tqdm(yt_graph.vs):

                yt_graph.vs[n.index]["label"] = mapping_labels[int(n["name"])]

                single_edgelist = [(e.index, e["weight"]) for e in yt_graph.es.select(_source=n)]

                tot = sum([w for _, w in single_edgelist])

                if tot > 0:

                    for eix, w in single_edgelist:

                        yt_graph.es[eix]["weight"] = w/tot   
                        
                        
            # save the graph
            with open(fn_graph, "wb") as f:

                pickle.dump(yt_graph, f)
                        
        
        else:
            
            with open(fn_graph, "rb") as f:

                yt_graph = pickle.load(f)
                        
                    
    

        # initialize 
        tau = 0.0
        one_instance = AbsorbingRandomWalk(yt_graph, label, d_, tau, mapping_scores, fn_f_matrix)

        mapping_id_names = {n.index : n["name"] for n in yt_graph.vs}
        one_instance._mapping_id_names =  mapping_id_names

        mapping_names_id = {n["name"]: n.index for n in yt_graph.vs}
        one_instance._mapping_names_id =  mapping_names_id                    
        
        one_instance.initialize()

        del one_instance