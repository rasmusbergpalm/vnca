import os

import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
from torch.utils.data import ConcatDataset

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


def load_model() -> VNCA:
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

    vnca = VNCA(
        h,
        w,
        n_channels,
        z_size,
        encoder,
        update_net,
        "hi",
        "hello",
        "hopefully this doesn't break",
        state_to_dist,
        batch_size,
        dmg_size,
    )
    print("loading the weights")
    vnca.load("latest")

    return vnca


def load_model_with_data():
    pass


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
