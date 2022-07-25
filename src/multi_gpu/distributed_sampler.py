import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.multiprocessing import spawn


class DummyDataset(Dataset):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx

def iterate(rank, loader):
    x = torch.cat([batch for batch in loader])
    x = [i.item() for i in x]
    print(f"{rank}: {x}")


def main(rank):
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "55554"
    import torch.distributed as dist
    dist.init_process_group(backend="nccl", init_method='env://', world_size=4, rank=rank)

    for size, batch_size in [
        (15, 2),
        (16, 16),
    ]:
        if rank == 0:
            print(f"size={size} batch_size={batch_size}")
            all_possiblities(rank, size, batch_size)

def all_possiblities(rank, size, batch_size):
    ds = DummyDataset(size=size)
    dist.barrier()
    if rank == 0: print("sampler=None drop_last=True")
    iterate(rank, DataLoader(ds, batch_size=batch_size, sampler=None, drop_last=True))
    dist.barrier()
    if rank == 0: print("sampler=None drop_last=False")
    iterate(rank, DataLoader(ds, batch_size=batch_size, sampler=None, drop_last=False))
    dist.barrier()
    if rank == 0: print("sampler=(drop_last=False) drop_last=False")
    iterate(rank, DataLoader(ds, batch_size=batch_size, sampler=DistributedSampler(ds, drop_last=False), drop_last=False))
    dist.barrier()
    if rank == 0: print("sampler=(drop_last=True) drop_last=False")
    iterate(rank, DataLoader(ds, batch_size=batch_size, sampler=DistributedSampler(ds, drop_last=True), drop_last=False))
    dist.barrier()
    if rank == 0: print("sampler=(drop_last=False) drop_last=True")
    iterate(rank, DataLoader(ds, batch_size=batch_size, sampler=DistributedSampler(ds, drop_last=True), drop_last=True))
    dist.barrier()
    if rank == 0: print("sampler=(drop_last=True) drop_last=True")
    iterate(rank, DataLoader(ds, batch_size=batch_size, sampler=DistributedSampler(ds, drop_last=True), drop_last=True))
    dist.barrier()

if __name__ == "__main__":
    spawn(main, nprocs=4)