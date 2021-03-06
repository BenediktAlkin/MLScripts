import os

import torch.utils.data as tdata
from ds import TestDataset
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

    def forward(self, x, verbose=False):
        if verbose:
            print(f"inner: {x.shape}")
        return self.main(x)

def get_optim(model):
    return optim.SGD(model.parameters(), lr=0.1)

def get_dist_sampler():
    with open("workload.yaml") as f:
        cfg = yaml.safe_load(f)
    ds = TestDataset(cfg)
    sampler = tdata.distributed.DistributedSampler(ds, shuffle=True, drop_last=False)
    return sampler

def train(model, optimizer, device=None, ds=None, rank=None):
    with open("workload.yaml") as f:
        cfg = yaml.safe_load(f)
    dims, batch_size, n_epochs = cfg["dims"], cfg["batch_size"], cfg["n_epochs"]
    ds = ds or TestDataset(cfg)
    if rank is not None:
        sampler = get_dist_sampler()
        # drop_last behavior:
        # sampler.drop_last=False --> if a device would not receive the full batch_size a duplicate is sampled
        loader = tdata.DataLoader(ds, shuffle=False, batch_size=batch_size, drop_last=False, sampler=sampler)
        set_epoch = sampler.set_epoch
    else:
        loader = tdata.DataLoader(ds, shuffle=True, batch_size=batch_size, drop_last=True)
        set_epoch = lambda _: None
    print(f"{rank} loader_len={len(loader)}")
    #print(f"{rank} {next(model.parameters())}")

    for i in range(n_epochs):
        set_epoch(i)
        for j, x in enumerate(loader):
            if device is not None:
                x = x.to(device)
            if i < 2:
                print(f"{rank} x: {[xi.item() for xi in x]}")
            if i == 0:
                #print(f"{rank} outer: {x.shape}")
                #y = model(x, verbose=True)
                y = model(x)
            else:
                y = model(x)
            model.zero_grad()
            y.mean().backward()
            optimizer.step()
