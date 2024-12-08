import pickle
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from graph_dataset import GraphDataset


def generate_dag(num_nodes, edge_prob, seed=None):
    if seed is not None:
        np.random.seed(seed)

    G = nx.DiGraph() 
    G.add_nodes_from(range(num_nodes+1))
    for i in range(0, num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_prob:
                G.add_edge(i, j)
    # Add a new tail node (num_nodes + 1) that connects to all current leaf nodes
    tail_node = num_nodes
    G.add_node(tail_node)
    for node in range(0,num_nodes):
        if G.out_degree(node) == 0:  # Check if the node has no children
            G.add_edge(node, tail_node)

    return G

def assign_labels(G, loc=0.0, scale=1.0):
    # Generate random feature vectors for each node
    feature_vectors = {node: np.random.normal(loc, scale, size=(64,)) for node in G.nodes}  # 10-dimensional feature vectors
    feature_vectors = {node: vec / np.linalg.norm(vec) for node, vec in feature_vectors.items()}
    nx.set_node_attributes(G, feature_vectors, 'features')

    # Generate random compute values for each node
    values = {node: np.random.uniform(0.1, 1) for node in G.nodes}
    nx.set_node_attributes(G, values, 'value')

    # Compute edge weights as the dot product between the corresponding features
    for u, v in G.edges:
        G[u][v]['weight'] = np.dot(G.nodes[u]['features'], G.nodes[v]['features'])
    
    # Aggregate features for each node based on the incoming edges
    for node in nx.topological_sort(G):
        #import pdb; pdb.set_trace()
        if G.in_degree(node) == 0:
            G.nodes[node]['y'] = G.nodes[node]['value']
        else:
            G.nodes[node]['y'] = G.nodes[node]['value'] + sum(G[u][v]['weight'] * G.nodes[u]['y'] for u, v in G.in_edges(node))
    

# def assign_errors(G):
#     errors = {}
#     for node in nx.topological_sort(G):
#         # intrinsic_error = np.random.normal(0, 1)
#         intrinsic_error = np.random.normal((G.nodes[node]["_C0"] / G.nodes[node]["compute"]) ** G.nodes[node]["_alpha"], 0.25)
#         parent_error = sum(G[u][v]['weight'] * errors[u] for u, v in G.in_edges(node))
#         noise = np.random.normal(0, 0.1)
#         errors[node] = intrinsic_error + parent_error + noise
#     return errors

def generate_dataset(num_graphs, num_nodes_minmax, edge_prob_minmax, **kwargs):
    graphs = []
    for graph_id in range(num_graphs):
        # Randomize num_nodes and edge_prob using a normal distribution
        num_nodes = np.random.randint(num_nodes_minmax[0], num_nodes_minmax[1]+1)
        edge_prob = np.random.uniform(edge_prob_minmax[0], edge_prob_minmax[1])
        
        G = generate_dag(num_nodes, edge_prob)
        assign_labels(G, **kwargs)
        visualize_graph(G)

        graphs.append(G)
    
    return graphs

def visualize_graph(G):
    # G = nx.DiGraph()
    # # Filter nodes and edges for the given graph_id
    # nodes = node_df[node_df['graph_id'] == graph_id]
    # edges = edge_df[edge_df['graph_id'] == graph_id]
    
    # # Add nodes and edges to the graph
    # G.add_nodes_from(nodes['node_id'].tolist())
    # for _, edge in edges.iterrows():
    #     G.add_edge(edge['source_node'], edge['target_node'], weight=edge['weight'])
    
    # Add layer attribute to nodes based on topological generations
    layers = {node: i for i, layer in enumerate(nx.topological_generations(G)) for node in layer}
    nx.set_node_attributes(G, layers, 'layer')

    # Prepare labels using the 'y' values and 'value' properties from the nodes
    labels = {node: f"y: {round(G.nodes[node]['y'], 2)}, value: {round(G.nodes[node]['value'], 2)}" for node in G.nodes}  # Round to 2 decimal places

    # Draw the graph with multipartite layout
    pos = nx.multipartite_layout(G, subset_key='layer')  # Use multipartite layout for visualization
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10, font_weight='bold', labels=labels)
    
    # Display edge weights
    labels = {edge: round(weight, 2) for edge, weight in nx.get_edge_attributes(G, 'weight').items()}  # Get edge weights and round to 2 decimal places
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)  # Draw edge labels
    plt.title(f"Graph Visualization")
    plt.show()


if __name__ == '__main__':
    graphs = generate_dataset(10000, (5, 15), (0.1, 0.3), loc=2.0, scale=4.0)
    test_size = 1000
    train_dataset = GraphDataset(graphs[:-test_size])

    # test_graphs = generate_dataset(100, (25, 40), (0.4, 0.8), loc=1.8, scale=1.0)
    test_dataset = GraphDataset(graphs[-test_size:])

    with open("train.pkl", "wb") as f:
        pickle.dump(train_dataset, f)

    with open("test.pkl", "wb") as f:
        pickle.dump(test_dataset, f)



