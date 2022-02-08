#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd

import csv
import igraph
import implicit
import glob


from tqdm import tqdm
from random import shuffle
from utils import get_sparse_adj_martrix


def generate_graph(filename, topk=False):
    
    input_edges = open(filename, "r")
    
    edges_reader = csv.reader(input_edges, delimiter="\t")
    
    mapping_outlinks = {}
    
    for (n1,n2, weight) in edges_reader:
        
        if n1 not in mapping_outlinks:
            
            mapping_outlinks[n1] = {}
        
        mapping_outlinks[n1][n2] = weight
    
    
    # generate top-k regular graph

    edgelist = []
    for source_node in mapping_outlinks:
        
        if topk:
            
            one_el = sorted(mapping_outlinks[source_node].items(), key = lambda x: x[1], reverse=True)
            
            if len(one_el) >= topk:
                
                edgelist += [(source_node, new_dest, int(weight)) for (new_dest, weight) in one_el][:topk]
        else:
            
            one_el = mapping_outlinks[source_node].items()#[:topk]
            
            edgelist += [(source_node, new_dest, int(weight)) for (new_dest, weight) in one_el]

    input_edges.close()
    
    # generate graph
    G_video = igraph.Graph.TupleList(edgelist, weights=True, directed=True)
    
    return G_video


def load_graph_and_category(filename, info_videos, topk):

    videograph = generate_graph(filename, topk)
    mapping_video_id_to_category = {vec[0]: vec[2].strip() for vec in info_videos.values}

    # big graph
    for n in videograph.vs:
        if n["name"] in mapping_video_id_to_category:
            n["category"] = mapping_video_id_to_category[n["name"]]
        else:
            n["category"] = "unknown"

    selected_nodes = [n for n in videograph.vs 
                      if n["category"] in ["Alt-lite", "Alt-right", "Intellectual Dark Web", "Media", "unknown"]
                     ]
    # selected-graph
    papergraph = videograph.subgraph(selected_nodes)
    for n in papergraph.vs:
        
        if n["category"] in ["Alt-lite", "Alt-right", "Intellectual Dark Web"]:
            
            n["category"] = "Bad"
    
    return papergraph



###############################################################################
# LOAD GRAPH
###############################################################################


# input-parameters
PATH = "../data/youtube/"
ITERATIONS = 10
FACTORS = 300
ALL_TOPK = [5,10,20]


filename = PATH + "no_sink_video_recommendations.tsv"

scores_out_fn = PATH + "final/yt-scores-distribution.tsv"


topk = False


info_videos = pd.read_csv(PATH + "videos.tsv", sep="\t")

initial_graph_with_weights = load_graph_and_category(filename, info_videos, topk)

initial_graph_with_weights.summary()


if glob.glob(scores_out_fn) == []:

    edgelist = [(e.source, e.target) for e in initial_graph_with_weights.es]

    edge_weights = [e["weight"] for e in initial_graph_with_weights.es]
    edge_weights = 1/(1+np.exp(-np.log2(edge_weights)))

    N = initial_graph_with_weights.vcount()

    #help(implicit.als.AlternatingLeastSquares)

    mm = get_sparse_adj_martrix(edgelist, weights=edge_weights, N=N)

    print("Training of ALS")

    model = implicit.als.AlternatingLeastSquares(factors=FACTORS,
                                                 calculate_training_loss=True, 
                                                 iterations=ITERATIONS,
                                                 #use_native=True
                                                )
    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(mm.T)

    all_nodes = [n.index for n in initial_graph_with_weights.vs]
    shuffle(all_nodes)

    upper_bound = 100



    new_edgelist = {topk: [] for topk in ALL_TOPK}


    # score-distribution

    scores_distribution = open(scores_out_fn, "w")

    writer_scores_distribution = csv.writer(scores_distribution, delimiter="\t")
    writer_scores_distribution.writerow(["node", "scores"])




    for source in tqdm(all_nodes):

        one_lst = model.recommend(source, user_items=mm, N=upper_bound)

        # write the scores
        writer_scores_distribution.writerow([source] + [(dest, round(score, 5)) for dest,score in one_lst])

        one_lst = [(source, new_dest, scores) for new_dest, scores in one_lst if source != new_dest]

        for topk in ALL_TOPK:        

            new_edgelist[topk] += one_lst[:topk]


    scores_distribution.close()

    
else:
    
    scores_distribution = open(scores_out_fn, "r")

    reader_scores_distribution = csv.reader(scores_distribution, delimiter="\t")
    
    header = next(reader_scores_distribution)
    
    new_edgelist = {topk: [] for topk in ALL_TOPK}
    
    for row in reader_scores_distribution:
        
        source = row[0]
        
        one_lst = [eval(t) for t in row[1:]]
        
        one_lst = [(source, new_dest, scores) for new_dest, scores in  one_lst]
        
        for topk in ALL_TOPK:
        
        
            new_edgelist[topk] += one_lst[:topk]
    


for topk in ALL_TOPK:

    df_new = pd.DataFrame(new_edgelist[topk], columns=["source","target","weight"])

    one_output_filename = PATH + "final/yt-top-%s-edges.tsv"%topk
    df_new.to_csv(one_output_filename, sep="\t", index=False)

# nodes
nodes_file = open(PATH + "final/yt-nodes.tsv", "w")

writer_nodes = csv.writer(nodes_file, delimiter="\t")
header_nodes = ["id", "name", "category"]
writer_nodes.writerow(header_nodes)

for n in initial_graph_with_weights.vs:
    writer_nodes.writerow([n.index, n["name"], n["category"]])

nodes_file.close()

