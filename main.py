import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Read the graph from edgelist
edgelist_path = "../dataset/musae_facebook_edges.csv"
G = nx.read_edgelist(edgelist_path, delimiter=',')

# Remove the self-loops from the graph
G.remove_edges_from(nx.selfloop_edges(G))

# Read the node information
info_path = "../dataset/musae_facebook_target.csv"
nodes_info = pd.read_csv(info_path)
nodes_info = nodes_info.drop(['facebook_id'], axis=1)

# One-hot encode the page type
page_type_one_hot = pd.get_dummies(nodes_info['page_type'], prefix='type')
nodes_info = nodes_info.drop(['page_type'], axis=1)

for column in page_type_one_hot.columns:
    nodes_info[column] = page_type_one_hot[column]
