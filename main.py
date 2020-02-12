import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Read the graph from edgelist
edge_list_path = "../dataset/musae_facebook_edges.csv"
G = nx.read_edgelist(edge_list_path, delimiter=',')

# Remove the self-loops from the graph
G.remove_edges_from(nx.selfloop_edges(G))
