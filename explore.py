import os
from typing import Tuple

import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
from torch.utils.data import ConcatDataset
from sklearn.manifold import TSNE

from modules.vnca import VNCA
from tasks.mnist.main import state_to_dist
from tasks.mnist.data import StaticMNIST


class LinearInterpolation:
    def __init__(self, n_points_in_line: int = 10):
        self.n_points_in_line = n_points_in_line

    def interpolate(self, z: t.Tensor, z_prime: t.Tensor) -> t.Tensor:
        zs = [
            (1 - t) * z + (t * z_prime) for t in t.linspace(0, 1, self.n_points_in_line)
        ]
        zs = t.vstack(zs)
        return zs

    def random_interpolation(self, zdim: int):
        z = t.randn((zdim))
        z_prime = t.randn((zdim))
        return self.interpolate(z, z_prime)


def get_binarized_MNIST_with_labels() -> Tuple[t.Tensor]:
    ims, labels = np.split(
        imageio.imread("https://i.imgur.com/j0SOfRW.png")[..., :3].ravel(), [-70000]
    )
    ims = np.unpackbits(ims).reshape((-1, 28, 28))
    return t.from_numpy(ims).type(t.float), t.from_numpy(labels)


def load_model(w_data: bool = False) -> VNCA:
    z_size = 128
    nca_hid = 128
    batch_size = 128
    dmg_size = 14

    filter_size = 5
    pad = filter_size // 2
    encoder_hid = 32
    h = w = 28
    n_channels = 1

    encoder = nn.Sequential(
        nn.Conv2d(n_channels, encoder_hid * 2 ** 0, filter_size, padding=pad + 2),
        nn.ELU(),  # (bs, 32, h, w)
        nn.Conv2d(
            encoder_hid * 2 ** 0,
            encoder_hid * 2 ** 1,
            filter_size,
            padding=pad,
            stride=2,
        ),
        nn.ELU(),  # (bs, 64, h//2, w//2)
        nn.Conv2d(
            encoder_hid * 2 ** 1,
            encoder_hid * 2 ** 2,
            filter_size,
            padding=pad,
            stride=2,
        ),
        nn.ELU(),  # (bs, 128, h//4, w//4)
        nn.Conv2d(
            encoder_hid * 2 ** 2,
            encoder_hid * 2 ** 3,
            filter_size,
            padding=pad,
            stride=2,
        ),
        nn.ELU(),  # (bs, 256, h//8, w//8)
        nn.Conv2d(
            encoder_hid * 2 ** 3,
            encoder_hid * 2 ** 4,
            filter_size,
            padding=pad,
            stride=2,
        ),
        nn.ELU(),  # (bs, 512, h//16, w//16),
        nn.Flatten(),  # (bs, 512*h//16*w//16)
        nn.Linear(encoder_hid * (2 ** 4) * 2 * 2, 2 * z_size),
    )

    update_net = nn.Sequential(
        nn.Conv2d(z_size, nca_hid, 3, padding=1),
        nn.ELU(),
        nn.Conv2d(nca_hid, z_size, 1, bias=False),
    )
    update_net[-1].weight.data.fill_(0.0)

    if w_data:
        data_dir = os.environ.get("DATA_DIR") or "data"
        train_data, val_data, test_data = (
            StaticMNIST(data_dir, "train"),
            StaticMNIST(data_dir, "val"),
            StaticMNIST(data_dir, "test"),
        )
        train_data = ConcatDataset((train_data, val_data))
    else:
        train_data = val_data = test_data = None

    vnca = VNCA(
        h,
        w,
        n_channels,
        z_size,
        encoder,
        update_net,
        train_data,
        test_data,
        test_data,
        state_to_dist,
        batch_size,
        dmg_size,
    )
    print("loading the weights")
    vnca.load("latest")

    return vnca


def get_imgs(z: t.Tensor, vnca: VNCA):
    seeds = (
        z.reshape((-1, vnca.z_size))  # stuff samples into batch dimension
        .unsqueeze(2)
        .unsqueeze(3)
        .expand(-1, -1, vnca.h, vnca.w)
    )
    f_states = vnca.decode(seeds)[-1]

    samples_imgs, p_imgs = vnca.to_rgb(f_states)
    return samples_imgs, p_imgs


def plot_random_interpolation():
    vnca = load_model()
    l = LinearInterpolation(25)
    zs = l.random_interpolation(128)

    _, axes = plt.subplots(5, 5)
    axes = axes.flatten()

    _, imgs = get_imgs(zs, vnca)

    for img, ax in zip(imgs, axes):
        ax.imshow(img[0])
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_interpolation_0_1():
    imgs, labels = get_binarized_MNIST_with_labels()
    ones_ = imgs[labels == 1]
    zeros_ = imgs[labels == 0]
    vnca = load_model()

    zs_ones = vnca.encode(ones_.unsqueeze(1)[:100]).mean
    zs_zeros = vnca.encode(zeros_.unsqueeze(1)[:100]).mean

    l = LinearInterpolation(25)
    zs = l.interpolate(zs_ones[0], zs_zeros[0])
    _, axes = plt.subplots(5, 5, figsize=(5 * 5, 5 * 5))
    axes = axes.flatten()

    _, imgs = get_imgs(zs, vnca)
    imgs = imgs.detach().numpy()

    for img, ax in zip(imgs, axes):
        ax.imshow(img[0])
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_clustering():
    imgs, labels = get_binarized_MNIST_with_labels()
    n = 5000
    vnca = load_model()
    encodings = vnca.encode(imgs[:n].unsqueeze(1)).mean.detach().numpy()
    tsne = TSNE(n_components=2)
    low_dim = tsne.fit_transform(encodings)
    _, ax = plt.subplots(1, 1)
    plot = ax.scatter(
        low_dim[:, 0],
        low_dim[:, 1],
        c=labels[:n].detach().numpy(),
        cmap="tab10",
        s=10,
        alpha=0.7,
    )
    plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")
    raise


if __name__ == "__main__":
    plot_clustering()
