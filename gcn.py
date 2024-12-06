import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
from rec_aggr_layer import RecursiveAggregationLayer

class GCNErrorPrediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNErrorPrediction, self).__init__()

        # GCN layers to learn node features
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.rec_aggr = RecursiveAggregationLayer()

        # Final prediction layer to compute total error
        # self.linear = nn.Linear(output_dim, 1)

    def forward(self, data):
        x, edge_index = data.x[:,1:], data.edge_index

        # Add self loops to ensure each node gets information from itself
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Apply GCN layers to update node features
        x = F.relu(self.conv1(x, edge_index))
        node_embeddings = F.relu(self.conv2(x, edge_index))

        edge_weights = self._calculate_edge_weights(node_embeddings, data.edge_index)

        output = self.rec_aggr(data.x[:,0], data.edge_index, edge_weights)
        # Compute predicted errors for each node
        # predicted_errors = self.linear(x)

        return output

    def _calculate_edge_weights(self, node_embeddings, edge_index):
        # Calculate edge weights as the dot product of parent and child node features
        edge_weights = []
        for edge in edge_index.t():
            source, target = edge[0], edge[1]
            # Compute the dot product between the feature vectors of the parent and child node
            edge_weight = torch.dot(node_embeddings[source], node_embeddings[target])
            edge_weights.append(edge_weight)
        return torch.stack(edge_weights)

# def calculate_aggregate(x, edge_index, edge_weights):
#     # Aggregate features for each node based on the incoming edges
#     aggregated_features = x.clone()
    
#     # Iterate over each edge and update the target node's features
#     for edge, weight in zip(edge_index.t(), edge_weights):
#         source, target = edge[0], edge[1]
#         aggregated_features[target] = aggregated_features[target] + (weight * aggregated_features[source])

#     return aggregated_features


# def calculate_aggregates(x, edge_index, edge_weights):
#     """
#     Aggregate node features based on incoming edges with edge weights.

#     Args:
#         x (torch.Tensor): Node feature matrix of shape [num_nodes, num_features].
#         edge_index (torch.LongTensor): Edge indices of shape [2, num_edges].
#         edge_weights (torch.Tensor): Edge weights of shape [num_edges].

#     Returns:
#         torch.Tensor: Aggregated node features of shape [num_nodes, num_features].
#     """
#     source_nodes = edge_index[0]  # Source nodes of edges
#     target_nodes = edge_index[1]  # Target nodes of edges

#     # Multiply source node features by edge weights
#     weighted_source = x[source_nodes] * edge_weights.unsqueeze(-1)  # Shape: [num_edges, num_features]
#     # Aggregate using scatter_add
#     aggregated = scatter_add(weighted_source, target_nodes, dim=0, dim_size=x.size(0)).sum(dim=1)
#     # Optionally, add the original node features (if needed)
#     aggregated += x
#     return aggregated


with open("graphs.pkl", "rb") as f:
    graphs = pickle.load(f)

# Model, optimizer, and criterion
input_dim = 10  # Number of features per node
hidden_dim = 16
output_dim = 10
model = GCNErrorPrediction(input_dim, hidden_dim, output_dim)

# Create a dataloader for graphs
train_loader = DataLoader(graphs, batch_size=2, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    losses = []
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        # data_embeddings = data.x[:,1:]

        # predictions = calculate_aggregate(data.x[:,0], data.edge_index, edge_weights)
        loss = criterion(output, data.y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    # if (epoch+1) % 100 == 0:
    print(f"Epoch {epoch+1}, Loss: {np.mean(losses)}")
    
