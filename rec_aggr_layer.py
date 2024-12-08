
import torch
import torch.nn as nn
from torch.autograd import Function


class RecursiveAggregationFunction(Function):
    @staticmethod
    def forward(ctx, x, edge_index, edge_weights):
        """
        Forward pass for recursive aggregation on DAGs.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, d].
            edge_index (torch.LongTensor): Edge indices of shape [2, num_edges].
            edge_weights (torch.Tensor): Edge weights of shape [num_edges].

        Returns:
            torch.Tensor: Aggregated node features of shape [num_nodes, d].
        """
        num_nodes = x.size(0)
        device = x.device

        # Perform topological sort to process nodes in order
        # topo_order = topological_sort(edge_index, num_nodes=num_nodes)
        topo_order = range(num_nodes)

        # Convert edge_index to lists for easier processing
        source_nodes = edge_index[0].tolist()
        target_nodes = edge_index[1].tolist()

        num_edges = edge_index.size(1)

        # Create adjacency list: for each node, list of (parent, edge_idx)
        adj = [[] for _ in range(num_nodes)]
        for edge_idx in range(num_edges):
            src = source_nodes[edge_idx]
            tgt = target_nodes[edge_idx]
            adj[tgt].append(edge_idx)

        # Initialize output as a clone of input features
        output = x.clone()

        # Iterate through nodes in topological order
        for node in topo_order:
            for edge_idx in adj[node]:
                parent = source_nodes[edge_idx]
                weight = edge_weights[edge_idx]
                # Perform weighted aggregation (no in-place operation)
                output[node] = output[node] + weight * output[parent]

        # Save tensors and structures needed for backward
        ctx.save_for_backward(edge_weights, torch.tensor(topo_order), output)
        ctx.num_nodes = num_nodes
        ctx.num_edges = num_edges
        ctx.source_nodes = torch.tensor(source_nodes, device=device)
        ctx.target_nodes = torch.tensor(target_nodes, device=device)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for recursive aggregation on DAGs.

        Args:
            grad_output (torch.Tensor): Gradient of loss w.r.t. output.

        Returns:
            Tuple[None, None, torch.Tensor]: Gradients w.r.t. inputs (x, edge_index, edge_weights).
        """
        edge_weights, topo_order, output = ctx.saved_tensors
        num_nodes = ctx.num_nodes
        num_edges = ctx.num_edges
        source_nodes = ctx.source_nodes
        target_nodes = ctx.target_nodes

        device = grad_output.device

        # Initialize gradient for edge_weights
        grad_edge_weights = torch.zeros(num_edges, device=device)

        # Initialize gradient for node outputs
        grad = grad_output.clone()

        # Create adjacency list: for each node, list of incoming edges
        adj = [[] for _ in range(num_nodes)]
        for edge_idx in range(num_edges):
            tgt = target_nodes[edge_idx].item()
            adj[tgt].append(edge_idx)

        # Traverse nodes in reverse topological order
        for node in reversed(topo_order.tolist()):
            for edge_idx in adj[node]:
                src = source_nodes[edge_idx]
                # Accumulate gradient for edge_weights

                grad_edge_weights[edge_idx] += torch.sum(grad[node] * output[src])
                # Propagate gradient to parent node
                grad[src] += edge_weights[edge_idx] * grad[node]

        # No gradients w.r.t. x and edge_index
        grad_x = None
        grad_edge_index = None

        return grad_x, grad_edge_index, grad_edge_weights


class RecursiveAggregationLayer(nn.Module):
    def __init__(self):
        super(RecursiveAggregationLayer, self).__init__()

    def forward(self, x, edge_index, edge_weights):
        return RecursiveAggregationFunction.apply(x, edge_index, edge_weights)

# Example Usage
if __name__ == "__main__":
    # Define a simple DAG
    # Example Graph:
    # 0 → 1 → 3
    # 0 → 2 → 3
    edge_index = torch.tensor([
        [0, 0, 1, 2],
        [1, 2, 3, 3]
    ], dtype=torch.long)

    # Node features (num_nodes=4, d=2)
    x = torch.tensor([
        [1.0, 2.0],  # Node 0
        [3.0, 4.0],  # Node 1
        [5.0, 6.0],  # Node 2
        [7.0, 8.0]   # Node 3
    ], requires_grad=False)  # Assuming node features do not require gradients

    # Edge weights (num_edges=4)
    edge_weights = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)

    # Initialize layer
    layer = RecursiveAggregationLayer()

    # Forward pass
    thing = layer(x, edge_index, edge_weights)
    print("Aggregated Output:\n", thing)

    # Example backward pass
    # Let's define a simple loss: sum of all output features
    loss = thing.sum()
    loss.backward()
    print("Gradients on edge_weights:\n", edge_weights.grad)
