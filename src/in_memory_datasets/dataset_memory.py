import functools

class DatasetMemory:
    def __init__(self, ds):
        self.ds = ds

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, item):
        return self.ds[item]

    def __len__(self):
        return len(self.ds)