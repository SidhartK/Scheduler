import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

class GCNErrorPrediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNErrorPrediction, self).__init__()

        # GCN layers to learn node features
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # Final prediction layer to compute total error
        self.linear = nn.Linear(output_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Add self loops to ensure each node gets information from itself
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Apply GCN layers to update node features
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Compute predicted errors for each node
        predicted_errors = self.linear(x)

        return predicted_errors

def calculate_edge_weights(x, edge_index):
    # Calculate edge weights as the dot product of parent and child node features
    edge_weights = []
    for edge in edge_index.t():
        source, target = edge[0].item(), edge[1].item()
        # Compute the dot product between the feature vectors of the parent and child node
        edge_weight = torch.dot(x[source], x[target])
        edge_weights.append(edge_weight)
    return torch.tensor(edge_weights)

def calculate_total_error(predicted_errors, edge_index, edge_weights):
    # Accumulate the intrinsic errors linearly from head to leaf by edge weights
    total_error = 0
    for edge, weight in zip(edge_index.t(), edge_weights):
        source, target = edge[0].item(), edge[1].item()
        total_error += predicted_errors[source] + predicted_errors[target] * weight
    return total_error

def train(model, data, optimizer, criterion, actual_error):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    predicted_errors = model(data)
    
    # Calculate the edge weights based on the feature vectors of the parent and child nodes
    edge_weights = calculate_edge_weights(data.x, data.edge_index)
    
    # Calculate the total error for the graph
    total_predicted_error = calculate_total_error(predicted_errors, data.edge_index, edge_weights)
    
    # MSE loss
    loss = criterion(total_predicted_error, actual_error)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Create a synthetic example of a DAG for demonstration
num_nodes = 5
node_features = torch.randn((num_nodes, 2))  # Random features

# Example edges (DAG)
edges = torch.tensor([
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 3],
    [3, 4]
], dtype=torch.long).t().contiguous()

# Actual total error for the graph
actual_error = torch.tensor([1.5], dtype=torch.float)

# Construct PyTorch Geometric Data object
data = Data(x=node_features, edge_index=edges)

# Model, optimizer, and criterion
input_dim = 2  # Number of features per node
hidden_dim = 16
output_dim = 8
model = GCNErrorPrediction(input_dim, hidden_dim, output_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    loss = train(model, data, optimizer, criterion, actual_error)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss}')
