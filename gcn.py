import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
# from torch_scatter import scatter_add
from tqdm import trange
from rec_aggr_layer import RecursiveAggregationLayer

class RAL(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RAL, self).__init__()

        # GCN layers to learn node features
        # self.conv1 = GCNConv(input_dim, hidden_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        # self.fc = nn.Sequential(
        #     nn.Linear(input_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, output_dim),
        # )
        # self.conv2 = GCNConv(hidden_dim, output_dim)
        self.rec_aggr = RecursiveAggregationLayer()

        # Final prediction layer to compute total error
        # self.linear = nn.Linear(output_dim, 1)

    def forward(self, data):
        x = data.x[:,1:]

        # Add self loops to ensure each node gets information from itself
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Apply GCN layers to update node features
        x = self.fc(x)
        # x = F.relu(self.conv1(x, edge_index))
        # x = F.relu(self.conv2(x, edge_index))

        edge_weights = self._calculate_edge_weights(x, data.edge_index)

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
    

class GCNBaseline(nn.Module):
    def __init__ (self, input_dim, hidden_dim, output_dim):
        super(GCNBaseline, self).__init__()

        self.convs1 = GCNConv(input_dim, hidden_dim)
        self.convs2 = GCNConv(hidden_dim, hidden_dim)
        self.convs3 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x[:,1:], data.edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        x = self.convs1(x, edge_index)
        x = F.relu(x)
        x = self.convs2(x, edge_index)
        x = F.relu(x)
        x = self.convs3(x, edge_index)
        return x


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


# with open("graphs.pkl", "rb") as f:
#     graphs = pickle.load(f)

# Model, optimizer, and criterion

baseline = GCNBaseline(64, 16, 1)
model = RAL(64, 16)

if os.path.exists("baseline.pth"):
    baseline.load_state_dict(torch.load("baseline.pth"))
if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth"))

if not (os.path.exists("baseline.pth") and os.path.exists("model.pth")):
    print("Training ...")
    # Create a dataloader for graphs
    with open("train.pkl", "rb") as f:
        graphs = pickle.load(f)
    train_loader = DataLoader(graphs, batch_size=32, shuffle=True)

    # Training loop
    baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    num_epochs = 10
    loop = trange(num_epochs, desc="Training")
    for epoch in loop:
        losses = []
        baseline_losses = []
        for data in train_loader:
            optimizer.zero_grad()
            baseline_optimizer.zero_grad()

            baseline_output = baseline(data)
            output = model(data)

            baseline_loss = criterion(baseline_output, data.y.unsqueeze(-1))
            baseline_losses.append(baseline_loss.item())
            baseline_loss.backward()

            loss = criterion(output, data.y)
            losses.append(loss.item())
            loss.backward()

            optimizer.step()
            baseline_optimizer.step()
        
        loop.set_postfix({"Base MSE": np.mean(baseline_losses), "Model MSE": np.mean(losses)})

    # Save the model
    torch.save(baseline.state_dict(), "baseline.pth")
    torch.save(model.state_dict(), "model.pth")

with open("test.pkl", "rb") as f:
    graphs = pickle.load(f)

# import pdb; pdb.set_trace()

test_loader = DataLoader(graphs, batch_size=1000, shuffle=False)
# Evaluation
baseline.eval()
with torch.no_grad():
    test_losses = []
    for data in test_loader:
        output = baseline(data)
        loss = criterion(output, data.y.unsqueeze(-1))
        test_losses.append(loss.item())
    print(f"Baseline Test Loss: {np.mean(test_losses)}")


model.eval()
with torch.no_grad():
    test_losses = []
    for data in test_loader:
        output = model(data)
        loss = criterion(output, data.y)
        test_losses.append(loss.item())
    print(f"Test Loss: {np.mean(test_losses)}")


