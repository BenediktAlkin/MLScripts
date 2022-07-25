import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.multiprocessing import spawn
import torch.distributed as dist


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
    gathered = [None for _ in range(dist.get_world_size())]
    dist.gather(x, gathered)
    gathered =torch.concat(gathered)
    print(f"{rank} all: {gathered}")


def main(rank):
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "55552"
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

# output
# sampler=None drop_last=True
# 2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# 3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# 1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# sampler=None drop_last=False
# 1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# 2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# 3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# sampler=(drop_last=False) drop_last=False
# 1: [1, 5, 9, 13]
# 2: [2, 6, 10, 14]
# 3: [3, 7, 11, 0]
# 0: [0, 4, 8, 12]
# sampler=(drop_last=True) drop_last=False
# 1: [1, 5, 9]
# 3: [3, 7, 11]
# 2: [2, 6, 10]
# 0: [0, 4, 8]
# sampler=(drop_last=False) drop_last=True
# 1: [1, 5, 9, 13]
# 2: [2, 6, 10, 14]
# 3: [3, 7, 11, 0]
# 0: [0, 4, 8, 12]
# sampler=(drop_last=True) drop_last=True
# 2: [2, 6]
# 3: [3, 7]
# 1: [1, 5]
# 0: [0, 4]