import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import community
import itertools
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Plot the degree distributions for different page types
def plot_degree_distributions():
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
    plt.xlim([0, 700])
    plt.subplot(222, title='Government')
    plt.hist(government_degrees, bins=100, log=True)
    plt.xlim([0, 700])
    plt.subplot(223, title='Company')
    plt.hist(company_degrees, bins=100, log=True)
    plt.xlim([0, 700])
    plt.subplot(224, title='Politician')
    plt.hist(politician_degrees, bins=100, log=True)
    plt.xlim([0, 700])
    plt.suptitle('Degree distributions')
    plt.subplots_adjust(hspace=0.4)
    plt.savefig('degree_distributions.png')
    plt.close()

# Check how many connected components there are and print them
def graph_n_components():
    graph_components = sorted(nx.connected_components(Graph), key = len, reverse=True)
    for component in graph_components:
        print("Number of nodes in component: ", len(component))
        print("-----------")

# Check the fractions of different page types
def page_type_fractions():
    nNodes = nodes_info.shape[0]
    for column in page_type_one_hot.columns:
        print(column, ': ', round(100 * nodes_info[column].sum() / nNodes, 1), '%')
    print("-----------")

# Compute the graph communities using the Louvain algorithm
def add_louvain_communities():
    partition = community.best_partition(Graph, random_state=42)
    nodes_info['community'] = -1
    for node in partition:
        nodes_info.iloc[int(node), nodes_info.columns.get_loc('community')] = int(partition.get(node))
    print("Community partition modularity: ", round(community.modularity(partition, Graph), 3))
    print("Number of communities: ", len(set(partition.values())))
    print("-----------")

# Save the node communities
def save_df_for_visualization():
        save_df = nodes_info[['id', 'page_type', 'community']].copy()
        save_df.to_csv('dataset/musae_facebook_target_visualization.csv', index=False)

def community_page_type_fractions():
    n_communities = nodes_info['community'].max() + 1
    tv_majority = 0
    government_majority = 0
    company_majority = 0
    politician_majority = 0
    for com in range(0, n_communities):
        community_df = nodes_info[nodes_info['community'] == com].copy()
        community_n_nodes = community_df.shape[0]
        print("Community size: ", community_n_nodes)

        tv_n = community_df['type_tvshow'].sum()
        gov_n = community_df['type_government'].sum()
        comp_n = community_df['type_company'].sum()
        pol_n = community_df['type_politician'].sum()

        types_n = [tv_n, gov_n, comp_n, pol_n]
        majority_n = types_n.index(max(types_n))
        if majority_n == 0:
            tv_majority = tv_majority + 1
            print("TV show majority")
        elif majority_n == 1:
            government_majority = government_majority + 1
            print("Government majority")
        elif majority_n == 2:
            company_majority = company_majority + 1
            print("Company majority")
        elif majority_n == 3:
            politician_majority = politician_majority + 1
            print("Politician majority")
        print("-----------")
    print('Fraction of communities with')
    print('Company majority: ',round(company_majority / n_communities, 2))
    print('Government majority: ',round(government_majority / n_communities, 2))
    print('Politician majority: ',round(politician_majority / n_communities, 2))
    print('TV show majority: ', round(tv_majority / n_communities, 2))
    print("-----------")

# Check if edges go from same type to same type, or different type to different type
def edge_types():
    n_edges = Graph.number_of_edges()
    print("Total number of edges: ", n_edges)
    total_same = 0
    total_diff = 0

    comp_same = 0
    comp_diff = 0
    gov_same = 0
    gov_diff = 0
    pol_same = 0
    pol_diff = 0
    tv_same = 0
    tv_diff = 0

    for edge in Graph.edges:
        node1 = int(edge[0])
        node2 = int(edge[1])
        ptype1 = nodes_info.iloc[node1, nodes_info.columns.get_loc('page_type')]
        ptype2 = nodes_info.iloc[node2, nodes_info.columns.get_loc('page_type')]
        if ptype1 == ptype2:
            total_same = total_same + 1
            if ptype1 == 'company':
                comp_same = comp_same + 1
            elif ptype1 == 'government':
                gov_same = gov_same + 1
            elif ptype1 == 'politician':
                pol_same = pol_same + 1
            elif ptype1 == 'tvshow':
                tv_same = tv_same + 1
        else:
            total_diff = total_diff + 1
            if (ptype1 or ptype2) == 'company':
                comp_diff = comp_diff + 1
            if (ptype1 or ptype2) == 'government':
                gov_diff = gov_diff + 1
            if (ptype1 or ptype2) == 'politician':
                pol_diff = pol_diff + 1
            if (ptype1 or ptype2) == 'tvshow':
                tv_diff = tv_diff + 1

    print("Edges from same-to-same to diff-to-diff page type")
    print("Total same: ", round(100 * total_same/n_edges, 1), "%")
    print("Total different: ", round(100 * total_diff/n_edges, 1), "%")
    print("-----------")
    print("Company same: ", round(100 * comp_same/(comp_same+comp_diff), 1), "%")
    print("Company different: ", round(100 * comp_diff/(comp_same+comp_diff), 1), "%")
    print("-----------")
    print("Government same: ", round(100 * gov_same/(gov_same+gov_diff), 1), "%")
    print("Government different: ", round(100 * gov_diff/(gov_same+gov_diff), 1), "%")
    print("-----------")
    print("Politician same: ", round(100 * pol_same/(pol_same+pol_diff), 1), "%")
    print("Politician different: ", round(100 * pol_diff/(pol_same+pol_diff), 1), "%")
    print("-----------")
    print("TV show same: ", round(100 * tv_same/(tv_same+tv_diff), 1), "%")
    print("TV show different: ", round(100 * tv_diff/(tv_same+tv_diff), 1), "%")
    print("-----------")

def load_data():
    print("Loading data...")
    with open("dataset/musae_facebook_features.json") as f:
        data = json.load(f)

    target = pd.read_csv("dataset/musae_facebook_target_visualization.csv")["page_type"]
    target_one_hot = pd.get_dummies(target) #one-hot encode target values

    featureMatrix = np.zeros((22470,4714), dtype=np.int8)
    for node_id in range(22470):
        for feature in range(4714):
            if feature in data[str(node_id)]:
                featureMatrix[node_id, feature] = 1

    featureMatrix = pd.DataFrame(featureMatrix)
    featureMatrix = featureMatrix.loc[:, (featureMatrix.sum(axis=0) > 1)] #remove features with only 1 node
    print("Data loaded.")
    return featureMatrix, target_one_hot

def randomForestAccuracy(featureMatrix, x_train, x_test, y_train, y_test, plot_importances = False, n_features=10):
    print("Training random forest...")
    #you can freely change the random forest hyperparameters below
    forest = RandomForestClassifier(n_estimators=100, random_state=1)
    forest.fit(x_train, y_train)
    print("Training done.")
    score = forest.score(x_test, y_test)
    print("Random forest accuracy: {:.2f} %".format(score*100))

    features = featureMatrix.columns
    importances = forest.feature_importances_
    #choose features that have relative importance higher than 0.001
    top_importances = importances[ importances > 0.001]
    indices = np.argsort(importances)[-len(top_importances):]
    #plot the importance of top n_features
    if plot_importances==True:
        #make another set of indices just for plotting
        indices_2 = np.argsort(importances)[-n_features:]
        plt.title('Top {} Feature Importances'.format(n_features))
        plt.barh(range(len(indices_2)), importances[indices_2], color='b', align='center')
        plt.yticks(range(len(indices_2)), [features[i] for i in indices_2])
        plt.xlabel('Relative Importance')
        plt.ylabel("Feature ID")
        plt.savefig("top{}_importances.png".format(n_features))
        plt.show()
        plt.close()

    return indices #return the indices of the most important features

def DNNaccuracy(x_train, x_test, y_train, y_test):
    #neural network topology is defined below
    classifier_inputs = layers.Input(shape=(x_train.shape[1],))
    classifier_hidden = layers.Dense(100, activation="relu")(classifier_inputs)
    classifier_hidden = layers.Dropout(0.4)(classifier_hidden)
    classifier_hidden = layers.Dense(100, activation="relu")(classifier_hidden)
    classifier_hidden = layers.Dropout(0.4)(classifier_hidden)
    classifier_hidden = layers.Dense(50, activation="relu")(classifier_hidden)
    classifier_out = layers.Dense(4, name="out_classifier", activation="softmax")(classifier_hidden)
    classifier_model = tf.keras.Model(inputs=[classifier_inputs], outputs=[classifier_out])

    opt_model = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    classifier_model.compile(loss="categorical_crossentropy", optimizer=opt_model, metrics=["accuracy"])

    classifier_model.fit(x_train, pd.get_dummies(y_train), epochs=100, batch_size=128, shuffle=True, validation_split=0.167)
    hist = classifier_model.evaluate(x_test, pd.get_dummies(y_test))
    print("DNN %s: %.2f%%" % (classifier_model.metrics_names[1], hist[1]*100))

if __name__ == '__main__':
    # Read the graph from edgelist
    edgelist_path = "dataset/musae_facebook_edges.csv"
    Graph = nx.read_edgelist(edgelist_path, delimiter=',')

    # Remove the self-loops from the graph
    Graph.remove_edges_from(nx.selfloop_edges(Graph))

    # Read the node information
    info_path = "dataset/musae_facebook_target.csv"
    nodes_info = pd.read_csv(info_path)
    nodes_info = nodes_info.drop(['facebook_id'], axis=1)

    # One-hot encode the page type
    page_type_one_hot = pd.get_dummies(nodes_info['page_type'], prefix='type')

    for column in page_type_one_hot.columns:
        nodes_info[column] = page_type_one_hot[column]

    ## Here it is possible to comment out processes
    plot_degree_distributions()
    graph_n_components()
    page_type_fractions()
    add_louvain_communities()
    community_page_type_fractions()
    save_df_for_visualization()
    edge_types()

    featureMatrix, target = load_data()
    #split data to training and test sets
    x_train, x_test, y_train, y_test = train_test_split(featureMatrix, target, test_size=0.4, random_state=1)
    #test the accuracy of a random forest and a DNN for node label prediction
    indices = randomForestAccuracy(featureMatrix, x_train, x_test, y_train, y_test, True)
    DNNaccuracy(x_train, x_test, y_train, y_test)

    #choose the most important features from the feature matrix and test again
    x_train = x_train.iloc[:,indices]
    x_test = x_test.iloc[:,indices]
    _ = randomForestAccuracy(featureMatrix, x_train, x_test, y_train, y_test)
    DNNaccuracy(x_train, x_test, y_train, y_test)
