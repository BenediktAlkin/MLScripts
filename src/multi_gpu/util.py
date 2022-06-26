import torch
import torch.nn as nn
import torch.optim as optim
import yaml

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        with open("workload.yaml") as f:
            cfg = yaml.safe_load(f)
        n_layers, dims = cfg["n_layers"], cfg["dims"]

        modules = []
        for _ in range(n_layers):
            modules.append(nn.Linear(dims, dims))
            modules.append(nn.ReLU())
        self.main = nn.Sequential(*modules)

    def forward(self, x):
        return self.main(x)

def get_optim(model):
    return optim.SGD(model.parameters(), lr=0.1)

def train(model, optimizer):
    with open("workload.yaml") as f:
        cfg = yaml.safe_load(f)
    dims, batch_size, n_batches = cfg["dims"], cfg["batch_size"], cfg["n_batches"]
    for _ in range(n_batches):
        x = torch.randn(batch_size, dims)
        y = model(x)
        model.zero_grad()
        y.mean().backward()
        optimizer.step()
