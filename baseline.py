from typing import Tuple

import imageio
import numpy as np
import torch as t
import torch.nn as nn
from torch import optim
from torch.distributions import Normal, Bernoulli, kl_divergence
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from shapeguard import ShapeGuard

from modules.iterable_dataset_wrapper import IterableWrapper
from modules.model import Model
from util import get_writers
from train import train


def get_binarized_MNIST_with_labels() -> Tuple[t.Tensor]:
    ims, labels = np.split(
        imageio.imread("https://i.imgur.com/j0SOfRW.png")[..., :3].ravel(), [-70000]
    )
    ims = np.unpackbits(ims).reshape((-1, 28, 28))
    return t.from_numpy(ims).type(t.float), t.from_numpy(labels)


class VAE(Model):
    def __init__(self, z_dim: int, batch_size: int):
        super().__init__()
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.z_dim = self.z_size = z_dim

        filter_size = 5
        pad = filter_size // 2
        encoder_hid = 32
        h = w = 28
        n_channels = 1
        self.h = h
        self.w = w
        self.encoder_hid = encoder_hid
        self.n_channels = n_channels

        self.encoder = nn.Sequential(
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
            nn.Linear(encoder_hid * (2 ** 4) * 2 * 2, 2 * self.z_size),
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(self.z_size, encoder_hid * (2 ** 4) * 2 * 2)
        )
        self.decoder = t.nn.Sequential(
            t.nn.ConvTranspose2d(
                encoder_hid * 2 ** 4,  # (bs, 512, h//16, w//16)
                encoder_hid * 2 ** 3,
                filter_size,
                padding=pad,
                output_padding=1,
                stride=2,
            ),
            t.nn.ELU(),
            nn.ConvTranspose2d(
                encoder_hid * 2 ** 3,
                encoder_hid * 2 ** 2,
                filter_size,
                padding=pad,
                output_padding=1,
                stride=2,
            ),
            t.nn.ELU(),
            nn.ConvTranspose2d(
                encoder_hid * 2 ** 2,
                encoder_hid * 2 ** 1,
                filter_size,
                padding=pad,
                output_padding=1,
                stride=2,
            ),
            t.nn.ELU(),
            nn.ConvTranspose2d(
                encoder_hid * 2 ** 1,
                encoder_hid * 2 ** 0,
                filter_size,
                padding=pad,
                output_padding=1,
                stride=2,
            ),
            t.nn.ELU(),
            nn.ConvTranspose2d(
                encoder_hid * 2 ** 0,
                n_channels,
                filter_size,
                padding=pad + 2,
                # output_padding=1,
            ),
        )

        data, labels = get_binarized_MNIST_with_labels()
        train_data = data[:50_000]
        train_labels = labels[:50_000]
        val_data = data[50_000:]
        val_labels = labels[50_000:]

        train_dataset = TensorDataset(train_data, train_labels)
        val_dataset = TensorDataset(val_data, val_labels)

        self.train_loader = iter(
            DataLoader(
                IterableWrapper(train_dataset), batch_size=batch_size, pin_memory=True
            )
        )
        self.val_loader = iter(
            DataLoader(
                IterableWrapper(val_dataset), batch_size=batch_size, pin_memory=True
            )
        )

        self.train_writer, self.test_writer = get_writers("baseline")

        self.p_z = Normal(
            loc=t.zeros((z_dim), device=self.device),
            scale=t.ones((z_dim), device=self.device),
        )
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.batch_idx = 0

    def encode(self, x: t.Tensor) -> Normal:
        mu_and_logsigma = self.encoder(x.to(self.device))
        mu = mu_and_logsigma[:, : self.z_dim]
        logsigma = mu_and_logsigma[:, self.z_dim :]

        return Normal(loc=mu, scale=t.exp(0.5 * logsigma))

    def decode(self, z: t.Tensor) -> Bernoulli:
        b, _ = z.shape
        res = self.decoder_linear(z.to(self.device)).view(
            b, self.encoder_hid * (2 ** 4), 2, 2
        )
        logits = self.decoder(res)

        return Bernoulli(logits=logits.view(-1, 28, 28))

    def forward(self, x: t.Tensor):
        ShapeGuard.reset()
        x.sg("bhw")
        x = x.unsqueeze(1).sg("bchw")
        q_z_given_x = self.encode(x)

        z = q_z_given_x.rsample()

        p_x_given_z = self.decode(z)

        return q_z_given_x, p_x_given_z

    def loss(self, x, q_z_given_x, p_x_given_z):
        rec_loss = -p_x_given_z.log_prob(x.to(self.device)).sum(dim=(1, 2))  # b
        kld = kl_divergence(self.p_z, q_z_given_x).sum(dim=1)  # b

        return (rec_loss + kld).mean()

    def eval_batch(self):
        self.train(False)
        with t.no_grad():
            x, _ = next(self.val_loader)
            q_z_given_x, p_x_given_z = self.forward(x)
            loss = self.loss(x, q_z_given_x, p_x_given_z)
            self.report(self.test_writer, loss)

        return loss.item()

    def train_batch(self):
        self.train(True)
        self.optimizer.zero_grad()
        x, _ = next(self.train_loader)
        q_z_given_x, p_x_given_z = self.forward(x)
        loss = self.loss(x, q_z_given_x, p_x_given_z)
        loss.backward()
        t.nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)

        self.optimizer.step()

        if self.batch_idx % 100 == 0:
            self.report(self.train_writer, loss)

        self.batch_idx += 1
        return loss

    def report(self, writer: SummaryWriter, loss):
        writer.add_scalar("loss", loss.mean().item(), self.batch_idx)
        writer.add_scalar(
            "bpd",
            loss.mean().item() / (np.log(2) * self.n_channels * self.h * self.w),
            self.batch_idx,
        )


if __name__ == "__main__":
    vae = VAE(128, 64)
    vae.eval_batch()
    train(vae, n_updates=100_000, eval_interval=50)
