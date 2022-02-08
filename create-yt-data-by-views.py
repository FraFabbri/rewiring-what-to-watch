
import pickle
import pandas as pd
import csv
from tqdm import tqdm


# input-parameters
lst_threshold = [10000, 100000]
lst_d_outdegree = [5, 10, 20]


### PATH - from config
### scores fn
PATH = "../data/youtube/"
scores_out_fn = PATH + "final/yt-scores-distribution.tsv"

nodes_df = pd.read_csv(PATH + "/final/yt-nodes.tsv", sep="\t")
nodes_df["category"] = nodes_df["category"].apply(lambda x: x.lower())


with open(scores_out_fn) as f: max_rows = len(f.readlines())

info_videos = pd.read_csv(PATH + "videos.tsv", sep="\t")


filtered_videos = {}

for threshold in lst_threshold:

    # subset of videos
    one_subset = set(info_videos[info_videos["view_count"] >= threshold]["video_id"])

    # subset of IDs
    filtered_videos[threshold] = set(nodes_df[nodes_df["name"].isin(one_subset)]["id"].apply(str))


scores_distribution = open(scores_out_fn, "r")


# reader
reader_scores_distribution = csv.reader(scores_distribution, delimiter="\t")
header = next(reader_scores_distribution)

# initialization
edgelist_by_threshold = {d: {t: [] for t in lst_threshold} for d in lst_d_outdegree}
mapping_scores_by_degree = {d: {t: {} for t in lst_threshold} for d in lst_d_outdegree}



# scroll rows
for _ in tqdm(range(max_rows-1)):

    row = next(reader_scores_distribution)

    source = row[0]
    out_ = [eval(t) for t in row[1:]]


    for t in lst_threshold:

        if source in filtered_videos[t]:

            out_filtered = [(str(source), str(dest), score) for (dest, score) in out_ if str(dest) in filtered_videos[t]]
            
            out_filtered = [(source, dest, score) for (source, dest, score) in out_filtered if source != dest]

            for d in lst_d_outdegree:

                    lst_edges = out_filtered[:d]

                    lst_recommendations = out_filtered[d:]


                    edgelist_by_threshold[d][t] += lst_edges
                    
                    mapping_scores_by_degree[d][t][str(source)] = lst_recommendations



with open(PATH + "final/edgelist-by-d-and-t.p", "wb") as f:

        pickle.dump(edgelist_by_threshold, f)


with open(PATH  + "final/scores-by-d-and-t.p", "wb") as f:

    pickle.dump(mapping_scores_by_degree, f)

