import random
from typing import Sequence, Tuple

import numpy as np
import torch as t
import torch.utils.data
import tqdm
from shapeguard import ShapeGuard
from torch import optim
from torch.distributions import Normal, Distribution, Bernoulli, kl_divergence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from modules.iterable_dataset_wrapper import IterableWrapper
from modules.loss import elbo, iwae
from modules.model import Model
from modules.dml import DiscretizedMixtureLogitsDistribution
from modules.nca import NCA
from util import get_writers


# torch.autograd.set_detect_anomaly(True)


class VAE(Model):
    def __init__(
        self,
        h: int,
        w: int,
        n_channels: int,
        z_size: int,
        encoder: t.nn.Module,
        decoder_linear: t.nn.Module,
        decoder: t.nn.Module,
        train_data: Dataset,
        val_data: Dataset,
        test_data: Dataset,
        batch_size: int,
        dmg_size: int,
        encoder_hid: int,
        n_mixtures: int = 1,
    ):
        super(Model, self).__init__()
        self.h = h
        self.w = w
        self.n_channels = n_channels
        self.z_size = z_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.n_damage = batch_size // 4
        self.dmg_size = dmg_size
        self.encoder_hid = encoder_hid
        self.n_mixtures = n_mixtures

        self.encoder = encoder
        self.decoder_linear = decoder_linear
        self.decoder = decoder
        self.p_z = Normal(
            t.zeros(self.z_size, device=self.device),
            t.ones(self.z_size, device=self.device),
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
        self.train_writer, self.test_writer = get_writers("baseline-celebA")

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
        x, y = next(self.train_loader)
        q_z_given_x, p_x_given_z = self.forward(x)
        loss = self.loss(x, q_z_given_x, p_x_given_z)
        loss.mean().backward()

        t.nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)

        self.optimizer.step()

        if self.batch_idx % 100 == 0:
            self.report(self.train_writer, loss)

        self.batch_idx += 1
        return loss.mean().item()

    def eval_batch(self):
        self.train(False)
        with t.no_grad():
            x, y = next(self.val_loader)
            q_z_given_x, p_x_given_z = self.forward(x)
            loss = self.loss(x, q_z_given_x, p_x_given_z)
            self.report(self.test_writer, loss)
        return loss.mean().item()

    def loss(self, x, q_z_given_x, p_x_given_z):
        rec_loss = -p_x_given_z.log_prob(x.to(self.device).squeeze(1)).sum(
            dim=(1, 2, 3)
        )  # b
        kld = kl_divergence(self.p_z, q_z_given_x).sum(dim=1)  # b

        return (rec_loss + kld).mean()

    def test(self, n_iw_samples):
        self.train(False)
        with t.no_grad():
            total_loss = 0.0
            for x, y in tqdm.tqdm(self.test_set):
                q_z_given_x, p_x_given_z = self.forward(x)
                loss = self.loss(x, q_z_given_x, p_x_given_z)
                total_loss += loss.mean().item()

        print(total_loss / len(self.test_set))

    def to_rgb(self, dist: DiscretizedMixtureLogitsDistribution):
        dist.sg("*chw")
        return dist.sample(), dist.mean

    def encode(self, x) -> Distribution:  # q(z|x)
        x.sg("Bchw")
        q = self.encoder(x).sg("BZ")
        loc = q[:, : self.z_size].sg("Bz")
        logsigma = q[:, self.z_size :].sg("Bz")
        return Normal(loc=loc, scale=t.exp(logsigma))

    def decode(self, z: t.Tensor) -> DiscretizedMixtureLogitsDistribution:  # p(x|z)
        z.sg("bz")
        b, _ = z.shape
        res = self.decoder_linear(z).view(
            b, self.encoder_hid * (2 ** 4), self.h // 16, self.w // 16
        )
        logits = self.decoder(res).sg(("b", self.n_mixtures * 10, "h", "w"))
        dist = DiscretizedMixtureLogitsDistribution(self.n_mixtures, logits)

        return dist

    def damage(self, hid: t.Tensor):
        b, _, _, w = hid.shape
        mask = t.ones_like(hid)
        for i in range(b):
            # h1 = random.randint(0, h - dmg_size)
            # w1 = random.randint(0, w - dmg_size)
            # mask[i, :, h1 : h1 + dmg_size, w1 : w1 + dmg_size] = 0.0
            mask[i, :, :, w // 2 :] = 0.0

        return hid * mask

    def damage_decode(
        self, z: t.Tensor, layer_pos: int
    ) -> DiscretizedMixtureLogitsDistribution:
        z.sg("bz")
        b, _ = z.shape
        res_undamaged = self.decoder_linear(z).view(
            b, self.encoder_hid * (2 ** 4), self.h // 16, self.w // 16
        )
        if layer_pos == 0:
            res_damaged = self.damage(res_undamaged)
            res = self.decoder(res_damaged)
        else:
            res_undamaged = self.decoder[:layer_pos](res_undamaged)
            res_damaged = self.damage(res_undamaged)
            res = self.decoder[layer_pos:](res_damaged)

        res.sg(("*", self.n_mixtures * 10, "h", "w"))
        dist = DiscretizedMixtureLogitsDistribution(self.n_mixtures, res)

        return dist

    def forward(self, x: t.Tensor):
        ShapeGuard.reset()
        x.sg("Bchw")
        x = x.to(self.device)

        q_z_given_x = self.encode(x).sg("Bz")
        z = q_z_given_x.rsample().sg("Bz")

        p_x_given_z = self.decode(z)

        return q_z_given_x, p_x_given_z

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
            p_x_given_z = self.decode(zs)
            samples = p_x_given_z.sample()
            samples_means = p_x_given_z.mean
            writer.add_images("samples/samples", samples, self.batch_idx)
            writer.add_images("samples/means", samples_means, self.batch_idx)

            # Reconstructions
            x, _ = next(self.val_loader)
            _, p_x_given_z = self.forward(x)
            recons_samples = p_x_given_z.sample()
            recons_means = p_x_given_z.mean

            writer.add_images("recons/samples", recons_samples, self.batch_idx)
            writer.add_images("recons/means", recons_means, self.batch_idx)

            # Damage
            # How do I visualize this?
