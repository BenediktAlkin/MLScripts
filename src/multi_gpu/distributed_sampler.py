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
    os.environ["MASTER_PORT"] = "55552"
    import torch.distributed as dist
    dist.init_process_group(backend="nccl", init_method='env://', world_size=4, rank=rank)
    ds = DummyDataset(size=15)
    batch_size = 2

    dist.barrier()
    if rank == 0: print("sampler=None drop_last=True")
    iterate(rank, DataLoader(ds, batch_size=batch_size, sampler=None, drop_last=True))
    dist.barrier()
    if rank == 0: print("sampler=None drop_last=False")
    iterate(rank, DataLoader(ds, batch_size=batch_size, sampler=None, drop_last=False))
    dist.barrier()
    if rank == 0: print("sampler=(drop_last=False) drop_last=False")
    sampler = DistributedSampler(ds, shuffle=False, drop_last=False)
    iterate(rank, DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=False))
    dist.barrier()
    if rank == 0: print("sampler=(drop_last=True) drop_last=False")
    sampler = DistributedSampler(ds, shuffle=False, drop_last=True)
    iterate(rank, DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=False))
    dist.barrier()
    if rank == 0: print("sampler=(drop_last=False) drop_last=True")
    sampler = DistributedSampler(ds, shuffle=False, drop_last=False)
    iterate(rank, DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=True))
    dist.barrier()
    if rank == 0: print("sampler=(drop_last=True) drop_last=True")
    sampler = DistributedSampler(ds, shuffle=False, drop_last=True)
    iterate(rank, DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=True))
    dist.barrier()

if __name__ == "__main__":
    spawn(main, nprocs=4)