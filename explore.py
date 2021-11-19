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

# TODO: change this.
from vaenca import VAENCA
from baseline import VAE


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


def get_all_static_data() -> t.Tensor:
    # train_imgs = np.loadtxt("mnist-static/binarized_mnist_train.amat")
    # test_imgs = np.loadtxt("mnist-static/binarized_mnist_test.amat")
    val_imgs = np.loadtxt("mnist-static/binarized_mnist_valid.amat")

    # all_imgs = np.concatenate((train_imgs, test_imgs, val_imgs))
    all_imgs = val_imgs
    return t.from_numpy(all_imgs).type(t.float).reshape(-1, 28, 28)


def load_model(w_data: bool = False) -> VAENCA:
    vnca = VAENCA(w_data)
    vnca.load("036578c")

    return vnca


def load_baseline() -> VAE:
    vae = VAE(load_data=False)
    print(vae.z_dim)
    vae.load("best_baseline_double")

    return vae


def get_imgs(z: t.Tensor, vnca: VAENCA):
    # TODO: implement this one. vnca.decode returns
    # a dist and the states.
    _, states = vnca.decode(z)
    state = states[-1]
    imgs = t.sigmoid(state[:, :1, :, :])

    return imgs


def plot_interpolation_0_1(model, name="vnca"):
    ShapeGuard.reset()
    imgs, labels = get_binarized_MNIST_with_labels()
    ones_ = imgs[labels == 1]
    zeros_ = imgs[labels == 0]

    zs_ones = model.encode(ones_.unsqueeze(1)[:100]).mean
    zs_zeros = model.encode(zeros_.unsqueeze(1)[:100]).mean

    l = LinearInterpolation(25)
    zs = l.interpolate(zs_ones[28], zs_zeros[-5])
    _, axes = plt.subplots(5, 5, figsize=(5 * 5, 5 * 5))
    axes = axes.flatten()

    ShapeGuard.reset()
    if "vnca" in name:
        imgs = get_imgs(zs.unsqueeze(1), model)
        # imgs = imgs[-1]
        imgs = imgs.detach().numpy()
    else:
        imgs = model.decode(zs).probs.detach().numpy()

    for img, ax in zip(imgs, axes):
        if "vnca" in name:
            ax.imshow(img[0])
        else:
            ax.imshow(img)
        ax.axis("off")

    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"./data/plots/interpolation_{name}.png", dpi=100)
    # plt.show()
    plt.close()


def plot_interpolation_vnca_doubling(
    model, digit_1=1, digit_2=0, idx_1=None, idx_2=None, name="doubling"
):
    ShapeGuard.reset()
    imgs, labels = get_binarized_MNIST_with_labels()
    imgs = t.nn.functional.pad(imgs, pad=[2, 2, 2, 2], mode="constant", value=0)
    ones_ = imgs[labels == digit_1]
    zeros_ = imgs[labels == digit_2]

    zs_ones = model.encode(ones_.unsqueeze(1)[:100]).mean
    zs_zeros = model.encode(zeros_.unsqueeze(1)[:100]).mean
    if idx_1 is None:
        idx_1 = np.random.randint(100)
    if idx_2 is None:
        idx_2 = np.random.randint(100)

    l = LinearInterpolation(8)
    zs = l.interpolate(zs_ones[idx_1], zs_zeros[idx_2])
    _, axes = plt.subplots(1, 8, figsize=(5 * 8, 5 * 1))
    axes = axes.flatten()

    ShapeGuard.reset()
    imgs = get_imgs(zs.unsqueeze(1), model).detach().numpy()

    for img, ax in zip(imgs, axes):
        ax.imshow(img[0], cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")

    # plt.title(name)
    plt.tight_layout()
    plt.savefig(
        f"./data/plots/final_model_{digit_1}_to_{digit_2}.png",
        dpi=100,
    )
    # plt.show()
    plt.close()


def plot_interpolation_baseline(
    model: VAE, digit_1=1, digit_2=0, idx_1=None, idx_2=None, name="baseline"
):
    ShapeGuard.reset()
    imgs, labels = get_binarized_MNIST_with_labels()
    imgs = t.nn.functional.pad(imgs, pad=[2, 2, 2, 2], mode="constant", value=0)

    ones_ = imgs[labels == digit_1]
    zeros_ = imgs[labels == digit_2]

    zs_ones = model.encode(ones_.unsqueeze(1)[:100]).mean
    zs_zeros = model.encode(zeros_.unsqueeze(1)[:100]).mean
    if idx_1 is None:
        idx_1 = np.random.randint(100)
    if idx_2 is None:
        idx_2 = np.random.randint(100)

    l = LinearInterpolation(8)
    zs = l.interpolate(zs_ones[idx_1], zs_zeros[idx_2])
    _, axes = plt.subplots(1, 8, figsize=(5 * 8, 5 * 1))
    axes = axes.flatten()

    ShapeGuard.reset()
    imgs = model.decode(zs).probs.detach().numpy()

    for img, ax in zip(imgs, axes):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")

    # plt.title(name)
    plt.tight_layout()
    plt.savefig(
        f"./data/plots/final_baseline_{digit_1}_to_{digit_2}.png",
        dpi=100,
    )
    # plt.show()
    plt.close()


def embed_static_data(model: VAENCA):
    """
    Just to check how the encodings look on the original dataset.
    Big shame they don't come with labels.
    """
    encodings = model.encode(model.val_data[:5000][0]).mean.detach().numpy()

    tsne = TSNE(n_components=2)
    low_dim = tsne.fit_transform(encodings)

    _, ax = plt.subplots(1, 1)
    ax.scatter(low_dim[:, 0], low_dim[:, 1])
    plt.show()
    plt.close()


def interpolate_at_random(model: VAENCA):
    imgs = get_all_static_data()

    img_1 = imgs[np.random.randint(len(imgs))]
    img_2 = imgs[np.random.randint(len(imgs))]

    z1 = model.encode(img_1.unsqueeze(0).unsqueeze(1)).mean
    z2 = model.encode(img_2.unsqueeze(0).unsqueeze(1)).mean

    l = LinearInterpolation(7)
    zs = l.interpolate(z1, z2)
    _, axes = plt.subplots(1, 7, figsize=(5 * 7, 5 * 1))
    axes = axes.flatten()

    ShapeGuard.reset()
    imgs = get_imgs(zs.unsqueeze(1), model).detach().numpy()

    for img, ax in zip(imgs, axes):
        ax.imshow(img[0], cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")

    # plt.title(name)
    plt.tight_layout()
    plt.savefig(f"./data/plots/doubling_random_interpolation.png", dpi=100)
    # plt.show()
    plt.close()


def plot_clustering(model, name="vnca", n=None, plot_colorbar=True):
    ShapeGuard.reset()
    imgs, labels = get_binarized_MNIST_with_labels()
    imgs = t.nn.functional.pad(imgs, pad=[2, 2, 2, 2], mode="constant", value=0)
    labels = labels.detach().numpy()
    if n is None:
        n = 5_000  # With all points is madness.
        print(f"Warning, computing a tSNE with {n} points")

    n_random_idxs = np.random.permutation(imgs.shape[0])[:n]

    # vnca = load_model()
    encodings = model.encode(imgs[n_random_idxs].unsqueeze(1)).mean.detach().numpy()
    tsne = TSNE(n_components=2)
    low_dim = tsne.fit_transform(encodings)
    _, ax = plt.subplots(1, 1)
    plot = ax.scatter(
        low_dim[:, 0],
        low_dim[:, 1],
        c=labels[n_random_idxs],
        cmap="tab10",
        s=10,
        alpha=0.7,
        vmin=np.min(labels) - 0.5,
        vmax=np.max(labels) + 0.5,
    )
    if plot_colorbar:
        plt.colorbar(
            plot,
            ax=ax,
            ticks=np.arange(np.min(labels), np.max(labels) + 1),
            fraction=0.046,
            pad=0.04,
        )
    ax.axis("off")
    # plt.title(name)
    plt.tight_layout()
    plt.savefig(f"./data/plots/clustering_{name}_{n}.png", dpi=100, bbox_inches="tight")
    plt.close()
    # raise


if __name__ == "__main__":
    model = load_model()
    baseline = load_baseline()

    # plot_clustering(model, name="vnca_doubling", plot_colorbar=False)
    # plot_clustering(baseline, name="baseline_DC")

    # embed_static_data(model)

    # plot_interpolation_0_1(model, name="vnca_doubling")
    # plot_interpolation_0_1(baseline, name="baseline_DC")
    # plot_interpolation_vnca_doubling(model, digit_1=1, digit_2=0, idx_1=99, idx_2=78)
    plot_interpolation_vnca_doubling(model, digit_1=2, digit_2=7)
    plot_interpolation_vnca_doubling(model, digit_1=5, digit_2=3)
    plot_interpolation_vnca_doubling(model, digit_1=4, digit_2=8)
    # plot_interpolation_baseline(baseline, digit_1=1, digit_2=0, idx_1=99, idx_2=78)
    # plot_interpolation_baseline(baseline, digit_1=2, digit_2=7)
    # plot_interpolation_baseline(baseline, digit_1=5, digit_2=3)
    # interpolate_at_random(model)


# put the tSNE of the baseline in the paper itself.
# Compare with damage on CelebA.
