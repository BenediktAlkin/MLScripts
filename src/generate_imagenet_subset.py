from torchvision.datasets import ImageFolder
import numpy as np
from pathlib import Path
import shutil

def main():
    main_impl("train")

def main_impl(split):
    keep_percentage = 0.01
    root = Path(f"C:/Users/Benedikt Alkin/Documents/data/ImageNetDebug/{split}")
    ds = ImageFolder(root)
    n_keep = int(len(ds) * keep_percentage)
    rng = np.random.default_rng(seed=5)
    idxs_to_keep = rng.choice(len(ds), n_keep, replace=False)
    for idx in idxs_to_keep:
        src_path, _ = ds.samples[idx]
        src_path = Path(src_path).relative_to(root)
        target_path = Path(f"{str(root.parent)}_{keep_percentage}") / split / src_path
        target_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(root / src_path, target_path)


if __name__ == "__main__":
    main()