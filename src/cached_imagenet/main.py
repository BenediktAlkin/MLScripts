import torch
from imagenet_cached import ImageNetCached
from torchvision.datasets import ImageFolder
from kappaprofiler import Stopwatch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def iterate_manually(ds, msg):
    with Stopwatch() as sw:
        for i in range(len(ds)):
            _ = ds[i]
    print(f"manual {msg}: {sw.elapsed_seconds}")

def iterate_loader(ds, msg, batch_size, num_workers, pin_memory, device):
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    with Stopwatch() as sw:
        for x, _ in loader:
            x.to(device)
    print(f"loader {msg} num_workers={num_workers} pin_memory={pin_memory}: {sw.elapsed_seconds}")

def benchmark_manual(root, transform):
    cached = ImageNetCached(root, transform=transform)
    #uncached = ImageFolder(root, transform=transform)

    iterate_manually(cached, "cached 1st")
    iterate_manually(cached, "cached 2nd")

def benchmark_loader(root, transform):
    device = torch.device("cuda:0")
    batch_size = 512
    for num_workers in [0, 1, 2]:
        for pin_memory in [False, True]:
            kwargs = dict(
                num_workers=num_workers,
                pin_memory=pin_memory,
                batch_size=batch_size,
                device=device,
            )
            cached = ImageNetCached(root, transform=transform)
            #uncached = ImageFolder(root, transform=transform)

            iterate_loader(cached, "cached 1st", **kwargs)
            iterate_loader(cached, "cached 2nd", **kwargs)
            #iterate_loader(uncached, "uncached 2nd", **kwargs)



def main():
    root = "C:/Users/Benedikt Alkin/Documents/data/ImageNetDebug/train"
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
    )
    benchmark_manual(root, transform)
    benchmark_loader(root, transform)



if __name__ == "__main__":
    main()