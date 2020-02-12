import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

edge_list_path = "../dataset/musae_facebook_edges.csv"
data_edges = pd.read_csv(edge_list_path)
