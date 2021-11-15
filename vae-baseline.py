import random
import os
from typing import Tuple, Sequence

import numpy as np
import torch as t
import tqdm
from torch.distributions import Distribution, Normal, Bernoulli, kl_divergence
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from shapeguard import ShapeGuard
from modules.model import Model
from modules.loss import elbo, iwae
from modules.iterable_dataset_wrapper import IterableWrapper
from util import get_writers
from tasks.mnist.data import StaticMNIST
from train import train


class BaselineVAE(Model):
    def __init__(
        self,
        h: int,
        w: int,
        n_channels: int,
        z_size: int,
        encoder: t.nn.Module,
        decoder: t.nn.Module,
        train_data: Dataset,
        val_data: Dataset,
        test_data: Dataset,
        batch_size: int,
    ):
        super(BaselineVAE, self).__init__()
        self.w = w
        self.h = h
        self.n_channels = n_channels
        self.z_size = z_size
        self.hidden_size = 256
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.train_loss_fn = self.elbo_loss
        self.test_loss_fn = self.elbo_loss

        self.encoder = encoder
        self.decoder = decoder

        self.p_z = Normal(
            t.zeros(self.z_size).to(self.device), t.ones(self.z_size).to(self.device)
        )

        self.test_set = test_data
        self.train_loader = iter(
            DataLoader(
                IterableWrapper(train_data), batch_size=batch_size, pin_memory=True
            )
        )
        self.val_loader = iter(
            DataLoader(
                IterableWrapper(val_data), batch_size=batch_size, pin_memory=True
            )
        )
        self.train_writer, self.test_writer = get_writers("baseline-vae")

        print(self)
        total = sum(p.numel() for p in self.parameters())
        for n, p in self.named_parameters():
            print(n, p.shape, p.numel(), "%.1f" % (p.numel() / total * 100))
        print("Total: %d" % total)

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.batch_idx = 0

    def train_batch(self):
        self.train(True)

        self.optimizer.zero_grad()
        x, _ = next(self.train_loader)
        loss, x, p_x_given_z, z, q_z_given_x = self.forward(x, 1, elbo)
        loss.mean().backward()

        t.nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)

        self.optimizer.step()

        if self.batch_idx % 100 == 0:
            self.report(self.train_writer, p_x_given_z, loss, None, None)

        self.batch_idx += 1
        return loss.mean().item()

    def eval_batch(self):
        self.train(False)
        with t.no_grad():
            x, _ = next(self.val_loader)
            loss, _, p_x_given_z, _, _ = self.forward(x, 1, iwae)
            self.report(self.test_writer, p_x_given_z, loss, None, None)
        return loss.mean().item()

    def test(self, n_iw_samples):
        self.train(False)
        with t.no_grad():
            total_loss = 0.0
            for x, y in tqdm.tqdm(self.test_set):
                loss, z, p_x_given_z, recon_loss, kl_loss, states = self.forward(
                    x, n_iw_samples, iwae
                )
                total_loss += loss.mean().item()

        print(total_loss / len(self.test_set))

    def to_rgb(self, state):
        dist: Distribution = self.state_to_dist(state)
        return dist.sample(), dist.mean

    def encode(self, x) -> Distribution:  # q(z|x)
        x.sg("Bchw")
        q = self.encoder(x).sg("BZ")
        loc = q[:, : self.z_size].sg("Bz")
        logsigma = q[:, self.z_size :].sg("Bz")
        return Normal(loc=loc, scale=t.exp(logsigma))

    def decode(self, z: t.Tensor) -> Bernoulli:  # p(x|z)
        z.sg("bz")
        res = self.decoder(z).view(-1, self.h, self.w)

        return Bernoulli(logits=res)

    def elbo_loss(
        self, x: t.Tensor, q_z_given_x: Distribution, p_x_given_z: Distribution
    ) -> t.Tensor:
        x.sg("bchw")
        x_ = x.squeeze(1).to(self.device).sg("bhw")  # assuming x is bchw.
        rec_loss = -p_x_given_z.log_prob(x_).sum(dim=(1, 2)).sg("b")  # b
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1).sg("b")  # b

        return (rec_loss + kld).mean()

    # def damage(self, states):
    #     states.sg("*zhw")
    #     mask = t.ones_like(states)
    #     for i in range(states.shape[0]):
    #         h1 = random.randint(0, states.shape[2] - self.dmg_size)
    #         w1 = random.randint(0, states.shape[3] - self.dmg_size)
    #         mask[i, :, h1 : h1 + self.dmg_size, w1 : w1 + self.dmg_size] = 0.0
    #     return states * mask

    def forward(self, x, n_samples, loss_fn):
        ShapeGuard.reset()
        x.sg("Bchw")
        x = x.to(self.device)
        q_z_given_x = self.encode(x).sg("Bz")
        z = q_z_given_x.rsample().sg("Bz")

        p_x_given_z = self.decode(z)
        loss = self.elbo_loss(x, q_z_given_x, p_x_given_z)

        return loss, x, p_x_given_z, z, q_z_given_x

    def report(self, writer: SummaryWriter, p_x_given_z, loss, recon_loss, kl_loss):
        writer.add_scalar("loss", loss.mean().item(), self.batch_idx)
        writer.add_scalar(
            "bpd",
            loss.mean().item() / (np.log(2) * self.n_channels * self.h * self.w),
            self.batch_idx,
        )

        ShapeGuard.reset()
        with t.no_grad():
            # samples
            samples = self.p_z.sample((64,))
            bern = self.decode(samples)
            images = bern.sample()
            writer.add_images("samples/samples", images.unsqueeze(1), self.batch_idx)
            writer.add_images("samples/means", bern.probs.unsqueeze(1), self.batch_idx)

            # Reconstructions
            x, _ = next(self.val_loader)
            # _, _, p_x_given_z, _, _ = self.forward(x[:64], 1, self.test_loss_fn)
            # recons_samples, recons_means = self.to_rgb(states[-1])
            # writer.add_images("recons/samples", recons_samples, self.batch_idx)
            # writer.add_images("recons/means", recons_means, self.batch_idx)


if __name__ == "__main__":
    z_size = 128
    batch_size = 128

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

    # A very normal decoder
    decoder = nn.Sequential(
        nn.Linear(z_size, 256),
        nn.ELU(),
        nn.Linear(256, 512),
        nn.ELU(),
        nn.Linear(512, n_channels * h * w),
    )

    data_dir = os.environ.get("DATA_DIR") or "data"
    train_data, val_data, test_data = (
        StaticMNIST(data_dir, "train"),
        StaticMNIST(data_dir, "val"),
        StaticMNIST(data_dir, "test"),
    )
    train_data = ConcatDataset((train_data, val_data))
    # train_data = val_data = test_data = None

    bvae = BaselineVAE(
        h,
        w,
        n_channels,
        z_size,
        encoder,
        decoder,
        train_data,
        test_data,
        test_data,
        batch_size,
    )

    bvae.eval_batch()
    train(bvae, n_updates=100_000, eval_interval=100)
