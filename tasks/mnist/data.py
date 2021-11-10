import os
import urllib

import certifi
import numpy as np
import torch as t
from torch.utils.data import Dataset


class StaticMNIST(Dataset):
    url_base = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
    splits = {
        'train': "binarized_mnist_train.amat",
        'val': "binarized_mnist_valid.amat",
        'test': "binarized_mnist_test.amat"
    }

    def __init__(self, data_dir, split):
        assert split in self.splits.keys()
        dir = data_dir + '/mnist-static'

        if not os.path.exists(dir):
            os.mkdir(dir)
            for file in self.splits.values():
                file_path = dir + '/' + file
                with urllib.request.urlopen(self.url_base + file, cafile=certifi.where()) as r:
                    with open(file_path, 'wb') as f:
                        f.write(r.read())

        file_path = dir + '/' + self.splits[split]

        with t.no_grad():
            self.samples = t.from_numpy(np.genfromtxt(file_path, delimiter=' ', dtype=np.float32)).reshape((-1, 1, 28, 28))

    def __getitem__(self, index):
        return self.samples[index], 0  # placeholder label

    def __len__(self):
        return len(self.samples)
