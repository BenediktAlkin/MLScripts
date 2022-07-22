from torchvision.datasets import ImageFolder
from functools import lru_cache

class ImageNetCached(ImageFolder):
    def __init__(self, root, transform=None, cached_transform=None):
        super().__init__(root, transform=transform)
        self.cached_transform = cached_transform

    @lru_cache(maxsize=None)
    def _get_sample(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.cached_transform is not None:
            sample = self.cached_transform(sample)
        return sample, target


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample, target = self._get_sample(index)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target