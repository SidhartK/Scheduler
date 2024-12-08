import torch
import torch.nn as nn
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