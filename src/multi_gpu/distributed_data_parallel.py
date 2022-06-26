import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import util
import kappaprofiler as kp
import os

def main():
    mp.spawn(main_mp, nprocs=4)


def main_mp(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "55555"
    print(f"rank: {rank}")
    dist.init_process_group(backend="nccl", init_method='env://', world_size=4, rank=rank)
    model = util.Model()
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = util.get_optim(model)
    with kp.Stopwatch() as sw:
        util.train(model, optimizer, device=device)
    print(sw.elapsed_seconds)


if __name__ == "__main__":
    main()