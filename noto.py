import os
import urllib.request
import zipfile
from glob import glob

import certifi
import torch as t
from PIL import Image
from torch.utils.data import Dataset, random_split


class NotoEmoji(Dataset):
    url = "https://www.dropbox.com/s/y6tlfrg0p634csj/noto-emoji-128.zip?dl=1"

    def __init__(self, data_dir, transform=lambda x: x):
        noto_dir = data_dir + '/noto-emoji-128'
        if not os.path.exists(noto_dir):
            noto_zip = data_dir + '/noto-emoji-128.zip'
            with urllib.request.urlopen(self.url, cafile=certifi.where()) as r:
                with open(noto_zip, 'wb') as f:
                    f.write(r.read())
            with zipfile.ZipFile(noto_zip, 'r') as zip_ref:
                zip_ref.extractall(noto_dir)

        self.samples = [(transform(Image.open(f)), 0) for f in glob(noto_dir + '/128/*.png')]

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def train_val_split(self, percent_val=0.2, seed=0):
        n_images = len(self)
        n_val = int(n_images * percent_val)
        return random_split(self, [n_images - n_val, n_val], generator=t.Generator().manual_seed(seed))


if __name__ == '__main__':
    NotoEmoji('/tmp')
