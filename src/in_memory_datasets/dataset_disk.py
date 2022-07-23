

class DatasetDisk:
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, item):
        return self.ds[item]

    def __len__(self):
        return len(self.ds)