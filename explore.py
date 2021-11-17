import os
from typing import Tuple

import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
from torch.utils.data import ConcatDataset
from sklearn.manifold import TSNE
from shapeguard import ShapeGuard

from modules.vnca import VNCA
from baseline import VAE
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
    vnca.load("latest_vnca")

    return vnca


def load_baseline() -> VAE:
    vae = VAE(128, 64)
    vae.load("best_baseline")

    return vae


def get_imgs(z: t.Tensor, vnca: VNCA):
    seeds = (
        z.reshape((-1, vnca.z_size))  # stuff samples into batch dimension
        .unsqueeze(2)
        .unsqueeze(3)
        .expand(-1, -1, vnca.h, vnca.w)
    )
    states = vnca.decode(seeds)
    # f_states = t.cat(f_states)
    samples_at_time_t = []
    p_at_time_t = []
    for state in states:
        samples_imgs, p_imgs = vnca.to_rgb(state)
        samples_at_time_t.append(samples_imgs)
        p_at_time_t.append(p_imgs)

    return samples_at_time_t, p_at_time_t


def plot_random_interpolation():
    vnca = load_model()
    l = LinearInterpolation(25)
    zs = l.random_interpolation(128)

    _, axes = plt.subplots(5, 5)
    axes = axes.flatten()

    _, imgs = get_imgs(zs, vnca)
    imgs = imgs[-1]

    for img, ax in zip(imgs, axes):
        ax.imshow(img[0])
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_interpolation_0_1(model, name="vnca"):
    ShapeGuard.reset()
    imgs, labels = get_binarized_MNIST_with_labels()
    ones_ = imgs[labels == 1]
    zeros_ = imgs[labels == 0]
    # vnca = load_model()

    # TODO: implement the interpolation using the baseline model.

    zs_ones = model.encode(ones_.unsqueeze(1)[:100]).mean
    zs_zeros = model.encode(zeros_.unsqueeze(1)[:100]).mean

    l = LinearInterpolation(25)
    zs = l.interpolate(zs_ones[0], zs_zeros[0])
    _, axes = plt.subplots(5, 5, figsize=(5 * 5, 5 * 5))
    axes = axes.flatten()

    if name == "vnca":
        _, imgs = get_imgs(zs, model)
        imgs = imgs[-1]
        imgs = imgs.detach().numpy()
    else:
        imgs = model.decode(zs).probs.detach().numpy()

    for img, ax in zip(imgs, axes):
        if name == "vnca":
            ax.imshow(img[0])
        else:
            ax.imshow(img)
        ax.axis("off")

    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"./data/plots/interpolation_{name}.png", dpi=100)
    # plt.show()
    plt.close()


def plot_clustering(model, name="vnca", n=5000):
    ShapeGuard.reset()
    imgs, labels = get_binarized_MNIST_with_labels()
    labels = labels.detach().numpy()
    # n = 5000
    # vnca = load_model()
    encodings = model.encode(imgs[:n].unsqueeze(1)).mean.detach().numpy()
    tsne = TSNE(n_components=2)
    low_dim = tsne.fit_transform(encodings)
    _, ax = plt.subplots(1, 1)
    plot = ax.scatter(
        low_dim[:, 0],
        low_dim[:, 1],
        c=labels[:n],
        cmap="tab10",
        s=10,
        alpha=0.7,
        vmin=np.min(labels) - 0.5,
        vmax=np.max(labels) + 0.5,
    )
    plt.colorbar(
        plot,
        ax=ax,
        ticks=np.arange(np.min(labels), np.max(labels) + 1),
        fraction=0.046,
        pad=0.04,
    )
    ax.axis("off")
    plt.title(name)
    plt.savefig(f"./data/plots/clustering_{name}.png")
    plt.close()
    # raise


if __name__ == "__main__":
    model = load_model()
    baseline = load_baseline()

    plot_clustering(model, name="vnca")
    plot_clustering(baseline, name="baseline_DC")

    plot_interpolation_0_1(model, name="vnca")
    plot_interpolation_0_1(baseline, name="baseline_DC")
