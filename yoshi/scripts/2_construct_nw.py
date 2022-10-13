import json
import os
import sys

import networkx as nx
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Input conf file")
        sys.exit(1)

    with open(sys.argv[1]) as fin:
        all_conf = json.load(fin)

    confname = all_conf['name']
    conf = all_conf['network_construction']

    input_filepath = conf["input_filepath"]
    customer_min_trans = conf['customer_min_trans']
    min_customer_num = conf['min_customer_num']
    yyyymm_periods = conf["yyyymm_periods"]

    ##
    df = pd.read_csv(input_filepath)

    # 1. Filter target period (month-basis)
    filtered_df = df[df["yyyymm"].isin(yyyymm_periods)]
    agg_df = filtered_df.groupby(['UYEISYERI_ID_MASK',
                                  'MUSTERI_ID_MASK']).agg(
                                      {'MUSTERI_ID_MASK': 'count'}).unstack()
    
    # 2. Create a customer list for each merchant
    mid_cs_list = []
    for index, row in agg_df.iterrows():
        cur_s = row[row >= customer_min_trans]
        if len(cur_s) > 0:
            customer_set = set(cur_s["MUSTERI_ID_MASK"].index.tolist())
        else:
            customer_set = set([])
        mid_cs_list.append([index, customer_set])

    # 3. Derive an edge between two nodes
    edge_list = []
    node_list = []
    for i in range(len(mid_cs_list)):
        mid_i, cs_i = mid_cs_list[i]
        node_list.append(mid_i)
        for j in range(i + 1, len(mid_cs_list)):
            mid_j, cs_j = mid_cs_list[j]
            count = len(cs_i & cs_j)
            if count > min_customer_num:
                # TODO: Do we want to consider the weight of the edge?
                edge_list.append((mid_i, mid_j))
                
    # 4. Construct graph
    g = nx.Graph()
    g.add_nodes_from(node_list)
    g.add_edges_from(edge_list)

    df_list = []
    for name, func, params in [("pr", nx.pagerank, {"alpha": 0.85}),
                               ("degree", nx.degree_centrality, {}),
                               ("closeness", nx.closeness_centrality, {}),
                               ("betweenness", nx.betweenness_centrality, {}),
                               ("eigenvector", nx.eigenvector_centrality, {})]:
                               # ("communicability_betweenness", nx.communicability_betweenness_centrality, {})]:
        print("{}...".format(name))
        c_dict = func(g, **params)
        c_df = pd.DataFrame(list(c_dict.items())).rename(
            columns={0: "mid", 1: name}).set_index("mid")
        df_list.append(c_df)
        print("done")
    concat_df = pd.concat(df_list, axis=1)
    concat_df.to_csv("output/{}_nw_features.csv".format(confname))
    with open("output/{}_stats.csv".format(confname), "w") as fout:
        fout.write(nx.info(g))

    ##
    output_dir = "fig/{}".format(confname)
    if not os.path.exists(output_dir):
        print("{} not exist. Create.".format(output_dir))
        os.makedirs(output_dir)
    
    for col in concat_df:
        concat_df[col].hist(bins=100)
        plt.savefig(os.path.join(output_dir,
                                 "{}_hist.png".format(col)))
        plt.close()

    nx.draw_circular(g, node_size=1, linewidths=0.5, alpha=0.5)
    plt.savefig(os.path.join(output_dir,
                             "all_nw_circular.png"))
    
    
