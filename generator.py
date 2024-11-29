import numpy as np
import networkx as nx
import pandas as pd

def generate_dag(num_nodes, edge_prob):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_prob:
                G.add_edge(i, j)
    return G

def assign_errors(G):
    errors = {}
    for node in nx.topological_sort(G):
        intrinsic_error = np.random.normal(0, 1)
        parent_error = sum(G[u][v]['weight'] * errors[u] for u, v in G.in_edges(node))
        noise = np.random.normal(0, 0.1)
        errors[node] = intrinsic_error + parent_error + noise
    return errors

def generate_dataset(num_graphs, num_nodes, edge_prob):
    data = []
    for _ in range(num_graphs):
        G = generate_dag(num_nodes, edge_prob)
        for u, v in G.edges:
            G[u][v]['weight'] = np.random.uniform(0.1, 1)
        errors = assign_errors(G)
        for node in G.nodes:
            data.append({
                "graph_id": _,
                "node_id": node,
                "node_degree": G.degree[node],
                "error": errors[node]
            })
    return pd.DataFrame(data)

# Generate a dataset
dataset = generate_dataset(100, 10, 0.3)
dataset.to_csv("dag_error_dataset.csv", index=False)
