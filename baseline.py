from typing import Tuple
import random
import os

import imageio
import numpy as np
import torch as t
import torch.nn as nn
from torch import optim
from torch.distributions import Normal, Bernoulli, kl_divergence
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from shapeguard import ShapeGuard

from iterable_dataset_wrapper import IterableWrapper
from modules.model import Model
from data.mnist import StaticMNIST
from torch.utils.data import ConcatDataset
from util import get_writers
from train import train


def get_binarized_MNIST_with_labels() -> Tuple[t.Tensor]:
    ims, labels = np.split(
        imageio.imread("https://i.imgur.com/j0SOfRW.png")[..., :3].ravel(), [-70000]
    )
    ims = np.unpackbits(ims).reshape((-1, 28, 28))
    return t.from_numpy(ims).type(t.float), t.from_numpy(labels)


class VAE(Model, nn.Module):
    def __init__(
        self,
        z_dim: int = 256,
        batch_size: int = 32,
        do_damage: bool = False,
        load_data: bool = True,
    ):
        super().__init__()
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.z_dim = self.z_size = z_dim

        filter_size = 5
        pad = filter_size // 2
        encoder_hid = 32
        # h = w = 28
        h = w = 32
        n_channels = 1
        self.h = h
        self.w = w
        self.encoder_hid = encoder_hid
        self.n_channels = n_channels
        self.do_damage = do_damage

        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, encoder_hid * 2 ** 0, filter_size, padding=pad),
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
                padding=pad,
                # output_padding=1,
            ),
        )

        self.deconv_positions = [
            i
            for i in range(len(self.decoder))
            if isinstance(self.decoder[i], nn.ConvTranspose2d)
        ]

        # data, labels = get_binarized_MNIST_with_labels()
        # train_data = data[:batch_size].unsqueeze(1)
        # train_labels = labels[:batch_size]
        # val_data = data[batch_size : 2 * batch_size].unsqueeze(1)
        # val_labels = labels[batch_size : 2 * batch_size]

        # train_data = TensorDataset(train_data, train_labels)
        # val_data = TensorDataset(val_data, val_labels)

        if load_data:
            data_dir = os.environ.get("DATA_DIR") or "data"
            train_data, val_data, test_data = (
                StaticMNIST(data_dir, "train"),
                StaticMNIST(data_dir, "val"),
                StaticMNIST(data_dir, "test"),
            )
            train_data = ConcatDataset((train_data, val_data))

            self.train_loader = iter(
                DataLoader(
                    IterableWrapper(train_data), batch_size=batch_size, pin_memory=True
                )
            )
            self.val_loader = iter(
                DataLoader(
                    IterableWrapper(test_data), batch_size=batch_size, pin_memory=True
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

        print(self)
        total = sum(p.numel() for p in self.parameters())
        for n, p in self.named_parameters():
            print(n, p.shape, p.numel(), "%.1f" % (p.numel() / total * 100))
        print("Total: %d" % total)

    def encode(self, x: t.Tensor) -> Normal:
        mu_and_logsigma = self.encoder(x.to(self.device))
        mu = mu_and_logsigma[:, : self.z_dim]
        logsigma = mu_and_logsigma[:, self.z_dim :]

        return Normal(loc=mu, scale=t.exp(0.5 * logsigma))

    def decode(self, z: t.Tensor, dmg: bool = None) -> Bernoulli:
        b, _ = z.shape
        res = self.decoder_linear(z.to(self.device)).view(
            b, self.encoder_hid * (2 ** 4), 2, 2
        )
        if dmg is None:
            do_damage = self.do_damage
        else:
            do_damage = dmg

        if do_damage:
            random_pos = random.choice(self.deconv_positions[:-1])
            res = self.decoder[:random_pos](res)
            res = self.damage(res)
            logits = self.decoder[random_pos:](res)
        else:
            logits = self.decoder(res)

        return Bernoulli(logits=logits.view(-1, self.h, self.w))

    def damage(self, res: t.Tensor):
        b, _, h, w = res.shape
        dmg_size = min(h // 2, w // 2)
        for i in range(b):
            h1 = random.randint(0, h - dmg_size)
            w1 = random.randint(0, w - dmg_size)
            res[i, :, h1 : h1 + dmg_size, w1 : w1 + dmg_size] = 0.0  # random damage.

        return res

    def forward(self, x: t.Tensor):
        ShapeGuard.reset()
        q_z_given_x = self.encode(x.sg("bchw"))

        z = q_z_given_x.rsample()

        p_x_given_z = self.decode(z)

        return q_z_given_x, p_x_given_z

    def loss(self, x, q_z_given_x, p_x_given_z):
        rec_loss = -p_x_given_z.log_prob(x.to(self.device).squeeze(1)).sum(
            dim=(1, 2)
        )  # b
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
        # Loss
        writer.add_scalar("loss", loss.mean().item(), self.batch_idx)
        writer.add_scalar(
            "bpd",
            loss.mean().item() / (np.log(2) * self.n_channels * self.h * self.w),
            self.batch_idx,
        )

        ShapeGuard.reset()
        with t.no_grad():
            # Samples
            zs = self.p_z.sample((64,))
            p_x_given_z = self.decode(zs, dmg=False)
            samples = p_x_given_z.sample().unsqueeze(1)
            samples_means = p_x_given_z.probs.unsqueeze(1)
            writer.add_images("samples/samples", samples, self.batch_idx)
            writer.add_images("samples/means", samples_means, self.batch_idx)

            # Reconstructions
            x, _ = next(self.val_loader)
            _, p_x_given_z = self.forward(x)
            recons_samples = p_x_given_z.sample().unsqueeze(1)
            recons_means = p_x_given_z.probs.unsqueeze(1)

            writer.add_images("recons/samples", recons_samples, self.batch_idx)
            writer.add_images("recons/means", recons_means, self.batch_idx)

            # Damage
            # How do I visualize this?

    def save(self, fn):
        t.save(
            {
                "batch_idx": self.batch_idx,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            fn,
        )

    def load(self, fn):
        checkpoint = t.load(fn, map_location=t.device(self.device))
        self.batch_idx = checkpoint["batch_idx"]
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


if __name__ == "__main__":
    vae = VAE(batch_size=128)
    vae.eval_batch()
    train(vae, n_updates=100_000, eval_interval=50)
    # raise
