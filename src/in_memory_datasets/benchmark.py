import torch
import kappaprofiler as kp
from dataset_memory import DatasetMemory
from dataset_disk import DatasetDisk
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

def main(ds_wrapper):
    ds = CIFAR10(root="~/Documents/data/CIFAR10", train=True, transform=ToTensor())
    wrapped_ds = ds_wrapper(ds)
    loader = DataLoader(wrapped_ds, batch_size=128, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)
    times = []
    with kp.Stopwatch() as sw:
        for _ in range(2):
            with kp.Stopwatch() as inner_sw:
                for x, _ in loader:
                    pass
            times.append(inner_sw.elapsed_seconds)
    print(f"{type(wrapped_ds).__name__}: {sw.elapsed_seconds} {times}")


if __name__ == "__main__":
    main(DatasetMemory)
    main(DatasetDisk)
    main(DatasetMemory)
    main(DatasetDisk)