import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

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
    node_data = []
    edge_data = []
    for graph_id in range(num_graphs):
        G = generate_dag(num_nodes, edge_prob)
        for u, v in G.edges:
            G[u][v]['weight'] = np.random.uniform(0.1, 1)
        errors = assign_errors(G)
        for node in G.nodes:
            node_data.append({
                "graph_id": graph_id,
                "node_id": node,
                "node_degree": G.degree[node],
                "error": errors[node]
            })
        for u, v in G.edges:
            edge_data.append({
                "graph_id": graph_id,
                "source_node": u,
                "target_node": v,
                "weight": G[u][v]['weight']
            })
    node_df = pd.DataFrame(node_data)
    edge_df = pd.DataFrame(edge_data)
    return node_df, edge_df

def visualize_graph(graph_id, node_df, edge_df):
    G = nx.DiGraph()
    # Filter nodes and edges for the given graph_id
    nodes = node_df[node_df['graph_id'] == graph_id]
    edges = edge_df[edge_df['graph_id'] == graph_id]
    
    # Add nodes and edges to the graph
    G.add_nodes_from(nodes['node_id'].tolist())
    for _, edge in edges.iterrows():
        G.add_edge(edge['source_node'], edge['target_node'], weight=edge['weight'])
    
    # Draw the graph
    pos = nx.spring_layout(G)  # Layout for visualization
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title(f"Graph ID: {graph_id}")
    plt.show()

# Generate dataset
node_dataset, edge_dataset = generate_dataset(100, 10, 0.3)
node_dataset.to_csv("dag_node_dataset.csv", index=False)
edge_dataset.to_csv("dag_edge_dataset.csv", index=False)

# Example usage to visualize a graph
visualize_graph(0, node_dataset, edge_dataset)
visualize_graph(1, node_dataset, edge_dataset)
visualize_graph(2, node_dataset, edge_dataset)
visualize_graph(3, node_dataset, edge_dataset)
