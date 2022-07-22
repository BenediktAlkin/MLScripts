import torch
from torch.utils.data import DataLoader
from kappaprofiler import Stopwatch
from imagenet_cached import ImageNetCached
import torchvision.transforms as transforms
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--device', type=str, required=True)
    return parser.parse_args()

def cpu_preprocessing1(root, device, batch_size, num_workers, pin_memory):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
    )
    ds = ImageNetCached(root, transform=transform)
    # load dataset into cache
    for i in range(len(ds)):
        _ = ds[i]

    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    with Stopwatch() as sw:
        for x, _ in loader:
            x.to(device)
    print(f"cpu_prep1 num_workers={num_workers} pin_memory={pin_memory}: {sw.elapsed_seconds}")

def cpu_preprocessing2(root, device, batch_size, num_workers, pin_memory):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
    )
    ds = ImageNetCached(root, transform=transform, cached_transform=transforms.ToTensor())
    # load dataset into cache
    for i in range(len(ds)):
        _ = ds[i]

    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    with Stopwatch() as sw:
        for x, _ in loader:
            x.to(device)
    print(f"cpu_prep2 num_workers={num_workers} pin_memory={pin_memory}: {sw.elapsed_seconds}")


def gpu_preprocessing1(root, device, batch_size, num_workers, pin_memory):
    cpu_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    ds = ImageNetCached(root, transform=cpu_transform)
    # load dataset into cache
    for i in range(len(ds)):
        _ = ds[i]

    gpu_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
    )

    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    with Stopwatch() as sw:
        for x, _ in loader:
            x = x.to(device)
            _ = gpu_transform(x)
    print(f"gpu_prep1 num_workers={num_workers} pin_memory={pin_memory}: {sw.elapsed_seconds}")


def gpu_preprocessing2(root, device, batch_size, num_workers, pin_memory):
    cached_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    ])
    ds = ImageNetCached(root, cached_transform=cached_transform)
    # load dataset into cache
    for i in range(len(ds)):
        _ = ds[i]

    gpu_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
    )

    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    with Stopwatch() as sw:
        for x, _ in loader:
            x = x.to(device)
            _ = gpu_transform(x)
    print(f"gpu_prep2 num_workers={num_workers} pin_memory={pin_memory}: {sw.elapsed_seconds}")

def benchmark(root, args, fn):
    device = torch.device(args.device)
    for num_workers in [0, 1]:
        for pin_memory in [False, True]:
            kwargs = dict(
                num_workers=num_workers,
                pin_memory=pin_memory,
                batch_size=args.batch_size,
                device=device,
            )
            fn(root, **kwargs)

def main():
    args = parse_args()
    root = "C:/Users/Benedikt Alkin/Documents/data/ImageNetDebug/train"
    benchmark(root, args, cpu_preprocessing1)
    benchmark(root, args, cpu_preprocessing2)
    benchmark(root, args, gpu_preprocessing1)
    benchmark(root, args, gpu_preprocessing2)



if __name__ == "__main__":
    main()