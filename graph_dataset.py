import torch
from torch_geometric.data import Data, Dataset
import numpy as np

class GraphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        G = self.graphs[idx]
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        x = torch.tensor(np.array([np.concatenate((np.array([G.nodes[node]['value']]), G.nodes[node]['features'])) for node in G.nodes]), dtype=torch.float)
        y = torch.tensor([G.nodes[node]['y'] for node in G.nodes], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y)