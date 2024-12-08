import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import trange

# Base class for models
class BaseModel(nn.Module):
    def forward(self, data):
        raise NotImplementedError("Forward method must be implemented.")

# Function to dynamically import model classes
def import_model_class(module_name, class_name):
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)

# List of model classes to train
model_classes = [
    #("ral_model", "RAL", {"input_dim": 64, "output_dim": 16}), 
    ("gcn_baseline_model", "GCNBaseline", {"input_dim": 64, "hidden_dim": 16,"output_dim": 1})
]

models = []
for module_name, class_name, params in model_classes:
    model_class = import_model_class(module_name, class_name)
    model_instance = model_class(**params)
    models.append(model_instance)

# Load models or initialize if not present
for model in models:
    model_name = model.__class__.__name__.lower()
    if os.path.exists(f"{model_name}.pth"):
        model.load_state_dict(torch.load(f"{model_name}.pth"))

criterion = nn.MSELoss()
# Training loop
if not all(os.path.exists(f"{model.__class__.__name__.lower()}.pth") for model in models):
    print("Training ...")
    with open("train.pkl", "rb") as f:
        graphs = pickle.load(f)
    train_loader = DataLoader(graphs, batch_size=32, shuffle=True)
    
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.01) for model in models]
    num_epochs = 10
    loop = trange(num_epochs, desc="Training")
    
    for epoch in loop:
        losses = []
        for data in train_loader:
            for model, optimizer in zip(models, optimizers):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, data.y)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
        
        loop.set_postfix({"MSE": np.mean(losses)})

    # Save models
    for model in models:
        torch.save(model.state_dict(), f"{model.__class__.__name__.lower()}.pth")

# Evaluation
with open("test.pkl", "rb") as f:
    graphs = pickle.load(f)

test_loader = DataLoader(graphs, batch_size=1000, shuffle=False)

for model in models:
    model.eval()
    with torch.no_grad():
        test_losses = []
        for data in test_loader:
            output = model(data)
            loss = criterion(output, data.y)
            test_losses.append(loss.item())
        print(f"{model.__class__.__name__} Test Loss: {np.mean(test_losses)}")


