import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Graph Construction
edges = torch.tensor([[0, 1], [1, 2], [1, 3], [2, 4], [3, 4]], dtype=torch.long).t()
node_features = torch.tensor([
    [10, 1],  # Task 0: Duration 10, Resource Requirement 1
    [20, 2],  # Task 1: Duration 20, Resource Requirement 2
    [15, 1],  # Task 2: Duration 15, Resource Requirement 1
    [10, 1],  # Task 3: Duration 10, Resource Requirement 1
    [5,  2]   # Task 4: Duration 5, Resource Requirement 2
], dtype=torch.float)

data = Data(x=node_features, edge_index=edges)
# y_data = Data(x=torch.tensor([
#     [10],
#     [30],
#     [45],
#     [40],
#     [50]
# ], dtype=torch.float))
y_data = Data(x=torch.tensor([
    [50]
], dtype=torch.float))

# Step 2: Define the GNN Model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        hidden_size = 2
        self.conv1 = GCNConv(data.num_node_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, hidden_size)
        self.conv4 = GCNConv(hidden_size, hidden_size)
        self.conv5 = GCNConv(hidden_size, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        return x

model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 3: Train the GNN
for epoch in range(8000):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out[-1], y_data.x[-1])
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f'Epoch {epoch}: Loss {loss.item()}')

# Step 4: Generate Schedule from GNN Output
def generate_schedule(task_durations, dependencies, resource_requirements):
    num_tasks = len(task_durations)
    start_times = np.zeros(num_tasks)
    finish_times = np.zeros(num_tasks)

    for i in range(num_tasks):
        if i == 0 or all(dependencies[:,1] != i):
            start_times[i] = 0  # Start the first task or tasks without dependencies immediately
        else:
            dependent_tasks = dependencies[:,0][dependencies[:,1] == i]
            start_times[i] = max([finish_times[j] for j in dependent_tasks])
        
        finish_times[i] = start_times[i] + task_durations[i]

    schedule = {f'Task {i}': (start_times[i], finish_times[i], resource_requirements[i]) for i in range(num_tasks)}
    return schedule

optimized_outputs = model(data).detach().numpy()
optimized_durations = optimized_outputs[:, 0]
print(optimized_durations)
print(F.mse_loss(torch.tensor(optimized_durations), y_data.x))
# resource_requirements = node_features[:, 1]
# schedule = generate_schedule(optimized_durations, edges.numpy().T, resource_requirements)

# Step 5: Visualize the Schedule
# fig, ax = plt.subplots()
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# for i, (task, (start, end, resource)) in enumerate(schedule.items()):
#     ax.plot([start, end], [i, i], marker='o', color=colors[i % len(colors)], label=task)
#     ax.text((start + end)/2, i, f'{task}\nResource: {resource:.2f}', 
#             horizontalalignment='center', verticalalignment='center')

# ax.set_yticks(range(len(schedule)))
# ax.set_yticklabels(schedule.keys())
# ax.set_title('Production Schedule')
# ax.set_xlabel('Time')
# plt.legend()
# plt.savefig("./schedule.png")