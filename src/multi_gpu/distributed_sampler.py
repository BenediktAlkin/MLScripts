import einops
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
    x = torch.cat([batch for batch in loader])#.to(torch.device(f"cuda:{rank}"))
    print(f"{rank}: {[i.item() for i in x]}")
    gathered = [torch.zeros_like(x)] * dist.get_world_size()
    dist.all_gather(gathered, x)
    # by default gathered is ordered by gpu, but samples are split round-robin
    # e.g. [0,1,2,3] with 2 gpus would be split into gpu0=[0,2] gpu1=[1,3] so concating would result in [0,2,1,3]
    gathered = torch.concat(gathered)
    print(f"{rank} all: {[i.item() for i in gathered]}")
    ordered = einops.rearrange(gathered, "(n_gpus x_size) -> (x_size n_gpus)", x_size=len(x))
    print(f"{rank} all ordered: {[i.item() for i in ordered]}")


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
# 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# 3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# 1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# 0 all: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# 3 all: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# 2 all: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# 1 all: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# 3 all ordered: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13]
# 0 all ordered: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13]
# 2 all ordered: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13]
# 1 all ordered: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13]
# sampler=None drop_last=False
# 3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# 1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# 2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# 3 all: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# 0 all: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# 1 all: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# 2 all: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# 3 all ordered: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14]
# 0 all ordered: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14]
# 1 all ordered: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14]
# 2 all ordered: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14]
# sampler=(drop_last=False) drop_last=False
# 3: [3, 7, 11, 0]
# 1: [1, 5, 9, 13]
# 0: [0, 4, 8, 12]
# 2: [2, 6, 10, 14]
# 3 all: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 0]
# 1 all: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 0]
# 0 all: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 0]
# 2 all: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 0]
# 3 all ordered: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0]
# 0 all ordered: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0]
# 1 all ordered: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0]
# 2 all ordered: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0]
# sampler=(drop_last=True) drop_last=False
# 3: [3, 7, 11]
# 1: [1, 5, 9]
# 0: [0, 4, 8]
# 2: [2, 6, 10]
# 3 all: [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
# 0 all: [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
# 1 all: [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
# 2 all: [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
# 3 all ordered: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# 0 all ordered: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# 1 all ordered: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# 2 all ordered: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# sampler=(drop_last=False) drop_last=True
# 3: [3, 7, 11, 0]
# 0: [0, 4, 8, 12]
# 1: [1, 5, 9, 13]
# 2: [2, 6, 10, 14]
# 3 all: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 0]
# 0 all: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 0]
# 1 all: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 0]
# 2 all: [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 0]
# 3 all ordered: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0]
# 0 all ordered: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0]
# 1 all ordered: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0]
# 2 all ordered: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0]
# sampler=(drop_last=True) drop_last=True
# 0: [0, 4]
# 3: [3, 7]
# 1: [1, 5]
# 2: [2, 6]
# 0 all: [0, 4, 1, 5, 2, 6, 3, 7]
# 3 all: [0, 4, 1, 5, 2, 6, 3, 7]
# 1 all: [0, 4, 1, 5, 2, 6, 3, 7]
# 2 all: [0, 4, 1, 5, 2, 6, 3, 7]
# 0 all ordered: [0, 1, 2, 3, 4, 5, 6, 7]
# 3 all ordered: [0, 1, 2, 3, 4, 5, 6, 7]
# 1 all ordered: [0, 1, 2, 3, 4, 5, 6, 7]
# 2 all ordered: [0, 1, 2, 3, 4, 5, 6, 7]