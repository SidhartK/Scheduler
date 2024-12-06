import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops

class GCNErrorPrediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNErrorPrediction, self).__init__()

        # GCN layers to learn node features
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # Final prediction layer to compute total error
        # self.linear = nn.Linear(output_dim, 1)

    def forward(self, data):
        x, edge_index = data.x[:,1:], data.edge_index

        # Add self loops to ensure each node gets information from itself
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Apply GCN layers to update node features
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Compute predicted errors for each node
        # predicted_errors = self.linear(x)

        return x

def calculate_edge_weights(x, edge_index):
    # Calculate edge weights as the dot product of parent and child node features
    edge_weights = []
    for edge in edge_index.t():
        source, target = edge[0], edge[1]
        # Compute the dot product between the feature vectors of the parent and child node
        edge_weight = torch.dot(x[source], x[target])
        edge_weights.append(edge_weight)
    return torch.stack(edge_weights)

def calculate_aggregate(x, edge_index, edge_weights):
    # Aggregate features for each node based on the incoming edges
    aggregated_features = x.clone()
    for edge, edge_weight in zip(edge_index.t(), edge_weights):
        source, target = edge[0].item(), edge[1].item()
        aggregated_features[target] += edge_weight * x[source] 
    return aggregated_features


# def accumulate_errors(x, edge_index, edge_weights, errors):
    # Compute the total error for each node based on the incoming edges
    

# # Create a synthetic example of a DAG for demonstration
# num_nodes = 5
# node_features = torch.randn((num_nodes, 2))  # Random features

# # Example edges (DAG)
# edges = torch.tensor([
#     [0, 1],
#     [0, 2],
#     [1, 3],
#     [2, 3],
#     [3, 4]
# ], dtype=torch.long).t().contiguous()

# # Actual total error for the graph
# actual_error = torch.tensor([1.5], dtype=torch.float)

# # Construct PyTorch Geometric Data object
# data = Data(x=node_features, edge_index=edges)

with open("graphs.pkl", "rb") as f:
    graphs = pickle.load(f)

# Model, optimizer, and criterion
input_dim = 10  # Number of features per node
hidden_dim = 16
output_dim = 10
# import pdb; pdb.set_trace()
model = GCNErrorPrediction(input_dim, hidden_dim, output_dim)

# Create a dataloader for graphs
train_loader = DataLoader(graphs, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    losses = []
    for data in train_loader:
        # optimizer.zero_grad()
        data_embeddings = model(data)
        edge_weights = calculate_edge_weights(data_embeddings, data.edge_index)
        predictions = calculate_aggregate(data.x[:,0], data.edge_index, edge_weights)

        loss = criterion(predictions, data.y)
        losses.append(loss.item())
        # loss.backward()
        # optimizer.step()
    
    # if (epoch+1) % 100 == 0:
    print(f"Epoch {epoch+1}, Loss: {np.mean(losses)}")
    
