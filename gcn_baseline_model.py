import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops
import torch.nn.functional as F

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
        return x.squeeze(-1)