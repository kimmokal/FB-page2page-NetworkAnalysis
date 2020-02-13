import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Read the graph from edgelist
edgelist_path = "../dataset/musae_facebook_edges.csv"
Graph = nx.read_edgelist(edgelist_path, delimiter=',')

# Remove the self-loops from the graph
Graph.remove_edges_from(nx.selfloop_edges(Graph))

# Read the node information
info_path = "../dataset/musae_facebook_target.csv"
nodes_info = pd.read_csv(info_path)
nodes_info = nodes_info.drop(['facebook_id'], axis=1)

# One-hot encode the page type for later use
page_type_one_hot = pd.get_dummies(nodes_info['page_type'], prefix='type')
nodes_info = nodes_info.drop(['page_type'], axis=1)

for column in page_type_one_hot.columns:
    nodes_info[column] = page_type_one_hot[column]

# Plot the degree distributions for different page types
tvshow_nodes = list(nodes_info['id'][nodes_info['type_tvshow'] == 1])
government_nodes = list(nodes_info['id'][nodes_info['type_government'] == 1])
company_nodes = list(nodes_info['id'][nodes_info['type_company'] == 1])
politician_nodes = list(nodes_info['id'][nodes_info['type_politician'] == 1])

degree_list = list(Graph.degree)
degree_dict = {}
for node, degree in degree_list:
    degree_dict.update({int(node) : degree})

tvshow_degrees = [degree_dict.get(node) for node in tvshow_nodes]
government_degrees = [degree_dict.get(node) for node in government_nodes]
company_degrees =  [degree_dict.get(node) for node in company_nodes]
politician_degrees =  [degree_dict.get(node) for node in politician_nodes]

plt.figure()
plt.subplot(221, title='TV show')
plt.hist(tvshow_degrees, bins=100, log=True)
plt.subplot(222, title='Government')
plt.hist(government_degrees, bins=100, log=True)
plt.subplot(223, title='Company')
plt.hist(company_degrees, bins=100, log=True)
plt.subplot(224, title='Politician')
plt.hist(politician_degrees, bins=100, log=True)
plt.suptitle('Degree distributions')
plt.subplots_adjust(hspace=0.4)
plt.savefig('degree_distributions.png')
plt.close()
