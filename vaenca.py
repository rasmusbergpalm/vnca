import os
from typing import Sequence, Tuple

import numpy as np
import torch as t
import torch.utils.data
from PIL import Image
from shapeguard import ShapeGuard
from torch import nn, optim
from torch.distributions import Normal, Distribution, kl_divergence, Bernoulli
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard._utils import make_grid
import tqdm

from data.mnist import StaticMNIST
from iterable_dataset_wrapper import IterableWrapper
from modules.model import Model
from modules.nca import MitosisNCA
from modules.residual import Residual
from train import train
from util import get_writers


# torch.autograd.set_detect_anomaly(True)


class VAENCA(Model, nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.h = self.w = 32
        self.z_size = 256
        self.train_loss_fn = self.elbo_loss_function
        self.train_samples = 1
        self.test_loss_fn = self.iwae_loss_fn
        self.test_samples = 1
        self.nca_hid = 256
        self.encoder_hid = 32
        batch_size = 32
        self.bpd_dimensions = 1 * 28 * 28
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        filter_size = (5, 5)
        pad = tuple(s // 2 for s in filter_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.encoder_hid * 2 ** 0, filter_size, padding=pad),
            nn.ELU(),  # (bs, 32, h, w)
            nn.Conv2d(
                self.encoder_hid * 2 ** 0,
                self.encoder_hid * 2 ** 1,
                filter_size,
                padding=pad,
                stride=2,
            ),
            nn.ELU(),  # (bs, 64, h//2, w//2)
            nn.Conv2d(
                self.encoder_hid * 2 ** 1,
                self.encoder_hid * 2 ** 2,
                filter_size,
                padding=pad,
                stride=2,
            ),
            nn.ELU(),  # (bs, 128, h//4, w//4)
            nn.Conv2d(
                self.encoder_hid * 2 ** 2,
                self.encoder_hid * 2 ** 3,
                filter_size,
                padding=pad,
                stride=2,
            ),
            nn.ELU(),  # (bs, 256, h//8, w//8)
            nn.Conv2d(
                self.encoder_hid * 2 ** 3,
                self.encoder_hid * 2 ** 4,
                filter_size,
                padding=pad,
                stride=2,
            ),
            nn.ELU(),  # (bs, 512, h//16, w//16),
            nn.Flatten(),  # (bs, 512*h//16*w//16)
            nn.Linear(
                self.encoder_hid * (2 ** 4) * self.h // 16 * self.w // 16,
                2 * self.z_size,
            ),
        )

        update_net = t.nn.Sequential(
            t.nn.Conv2d(self.z_size, self.nca_hid, 3, padding=1),
            Residual(
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
                t.nn.ELU(),
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
            ),
            Residual(
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
                t.nn.ELU(),
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
            ),
            Residual(
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
                t.nn.ELU(),
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
            ),
            Residual(
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
                t.nn.ELU(),
                t.nn.Conv2d(self.nca_hid, self.nca_hid, 1),
            ),
            t.nn.Conv2d(self.nca_hid, self.z_size, 1),
        )
        update_net[-1].weight.data.fill_(0.0)
        update_net[-1].bias.data.fill_(0.0)

        self.nca = MitosisNCA(
            self.h, self.w, self.z_size, update_net, int(np.log2(self.h)) - 1, 8, 1.0
        )
        self.p_z = Normal(
            t.zeros(self.z_size, device=self.device),
            t.ones(self.z_size, device=self.device),
        )

        # data_dir = os.environ.get('DATA_DIR') or "."
        # train_data, val_data = StaticMNIST(data_dir, 'train'), StaticMNIST(data_dir, 'val'),
        # train_data = ConcatDataset((train_data, val_data))
        # self.test_set = StaticMNIST(data_dir, 'test')
        # self.train_loader = iter(DataLoader(IterableWrapper(train_data), batch_size=batch_size, pin_memory=True))
        # self.test_loader = iter(DataLoader(IterableWrapper(self.test_set), batch_size=batch_size, pin_memory=True))
        # self.train_writer, self.test_writer = get_writers("hierarchical-nca")

        print(self)
        for n, p in self.named_parameters():
            print(n, p.shape)

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.batch_idx = 0

    def train_batch(self):
        self.train(True)

        self.optimizer.zero_grad()
        x, y = next(self.train_loader)
        loss, z, p_x_given_z, recon_loss, kl_loss = self.forward(
            x, self.train_samples, self.train_loss_fn
        )
        loss.backward()

        t.nn.utils.clip_grad_norm_(self.parameters(), 10.0)

        self.optimizer.step()

        if self.batch_idx % 100 == 0:
            self.report(self.train_writer, p_x_given_z, loss, recon_loss, kl_loss)

        self.batch_idx += 1
        return loss.item()

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

    def eval_batch(self):
        self.train(False)
        with t.no_grad():
            x, y = next(self.test_loader)
            loss, z, p_x_given_z, recon_loss, kl_loss = self.forward(
                x, self.test_samples, self.test_loss_fn
            )
            self.report(self.test_writer, p_x_given_z, loss, recon_loss, kl_loss)
        return loss.item()

    def test(self, n_iw_samples):
        self.train(False)
        with t.no_grad():
            total_loss = 0.0
            for x, y in tqdm.tqdm(self.test_set.samples):
                loss, z, p_x_given_z, recon_loss, kl_loss = self.forward(
                    x, n_iw_samples, self.test_loss_fn
                )
                total_loss += loss

        print(total_loss / len(self.test_set))

    def _plot_samples(self):
        ShapeGuard.reset()
        with torch.no_grad():
            samples = self.p_z.sample((64, 1)).to(self.device)
            decode, states = self.decode(samples)
            samples = self.to_rgb(states[-1])
            # rgb=0.3, alpha=0 --> samples = 1-0+0.3*0 = 1 = white
            # rgb = 0.3, alpha=1 --> samples = 1-1+0.3*1 = 0.3
            # rgb = 0.3, alpha=0.5, samples = 1-0.5+0.3*0.5 = 0.5+0.15 = 0.65

            growth = []
            for state in states:
                rgb = self.to_rgb(state[0:1])
                h = state.shape[3]
                pad = (self.h - h) // 2
                rgb = t.nn.functional.pad(rgb, [pad] * 4, mode="constant", value=0)
                growth.append(rgb)
            growth = t.cat(growth, dim=0).cpu().detach().numpy()  # (n_states, 3, h, w)

            x, y = next(self.test_loader)
            _, _, p_x_given_z, _, _ = self.forward(x[:64], 1, self.iwae_loss_fn)
            recons = self.to_rgb(p_x_given_z.logits.reshape(-1, 1, self.h, self.w))

        return samples, recons, growth

    def to_rgb(self, samples):
        return Bernoulli(logits=samples[:, :1, :, :]).sample()

    def plot_growth_samples(self):
        ShapeGuard.reset()
        with torch.no_grad():
            samples = self.p_z.sample((64, 1)).to(self.device)
            _, states = self.decode(samples)
            for i, state in enumerate(states):
                samples = t.clip(state[:, :4, :, :], 0, 1).cpu().detach().numpy()
                samples = self.to_rgb(samples)
                samples = (samples * 255).astype(np.uint8)
                grid = make_grid(samples).transpose(1, 2, 0)  # (HWC)
                im = Image.fromarray(grid)
                im.save("samples-%03d.png" % i)

    def report(self, writer: SummaryWriter, p_x_given_z, loss, recon_loss, kl_loss):
        writer.add_scalar("loss", loss.item(), self.batch_idx)
        writer.add_scalar(
            "bpd", loss.item() / (np.log(2) * self.bpd_dimensions), self.batch_idx
        )
        writer.add_scalar(
            "entropy", p_x_given_z.entropy().mean().item(), self.batch_idx
        )
        if recon_loss:
            writer.add_scalar("recon_loss", recon_loss.item(), self.batch_idx)
        if kl_loss:
            writer.add_scalar("kl_loss", kl_loss.item(), self.batch_idx)

        samples, recons, growth = self._plot_samples()
        # writer.add_images("grid", grid, self.batch_idx)
        writer.add_images("samples", samples, self.batch_idx)
        writer.add_images("recons", recons, self.batch_idx)
        writer.add_images("growth", growth, self.batch_idx)

    def encode(self, x) -> Distribution:  # q(z|x)
        x.sg("B4hw")
        q = self.encoder(x).sg("BZ")
        loc = q[:, : self.z_size].sg("Bz")
        logsigma = q[:, self.z_size :].sg("Bz")
        return Normal(loc=loc, scale=t.exp(logsigma))

    def decode(self, z: t.Tensor) -> Tuple[Distribution, Sequence[t.Tensor]]:  # p(x|z)
        z.sg("Bnz")
        bs, ns, zs = z.shape
        state = (
            z.reshape((-1, self.z_size))
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, 2, 2)
            .sg("bz22")
        )
        states = self.nca(state)

        state = states[-1]

        logits = state[:, :1, :, :].sg("b1hw").reshape((bs, ns, -1)).sg("Bnx")

        return Bernoulli(logits=logits), states

    def forward(self, x, n_samples, loss_fn):
        ShapeGuard.reset()
        x.sg("B4hw")
        x = x.to(self.device)
        x_flat = x.reshape(x.shape[0], -1).sg("Bx")
        q_z_given_x = self.encode(x).sg("Bz")
        z = q_z_given_x.rsample((n_samples,)).permute((1, 0, 2)).sg("Bnz")
        decode, _ = self.decode(z)
        p_x_given_z = decode.sg("Bnx")

        loss, recon_loss, kl_loss = loss_fn(x_flat, p_x_given_z, q_z_given_x, z)
        return loss, z, p_x_given_z, recon_loss, kl_loss

    def iwae_loss_fn(
        self,
        x: t.Tensor,
        p_x_given_z: Distribution,
        q_z_given_x: Distribution,
        z: t.Tensor,
    ):
        """
        log(p(x)) >= logsumexp_{i=1}^N[ log(p(x|z_i)) + log(p(z_i)) - log(q(z_i|x))] - log(N)
        """
        x.sg("Bx")
        p_x_given_z.sg("Bnx")
        q_z_given_x.sg("Bz")
        z.sg("Bnz")

        logpx_given_z = (
            p_x_given_z.log_prob(x.unsqueeze(1).expand_as(p_x_given_z.mean))
            .sum(dim=2)
            .sg("Bn")
        )
        logpz = self.p_z.log_prob(z).sum(dim=2).sg("Bn")
        logqz_given_x = (
            q_z_given_x.log_prob(z.permute((1, 0, 2)))
            .sum(dim=2)
            .permute((1, 0))
            .sg("Bn")
        )
        logpx = (
            t.logsumexp(logpx_given_z + logpz - logqz_given_x, dim=1)
            - t.log(t.scalar_tensor(z.shape[1]))
        ).sg("B")
        return -logpx.mean(), None, None  # (1,)

    def elbo_loss_function(
        self,
        x: t.Tensor,
        p_x_given_z: Distribution,
        q_z_given_x: Distribution,
        z: t.Tensor,
    ):
        """
        log p(x) >= E_q(z|x) [ log p(x|z) p(z) / q(z|x) ]
        Reconstruction + KL divergence losses summed over all elements and batch
        """
        x.sg("Bx")
        p_x_given_z.sg("Bnx")
        q_z_given_x.sg("Bz")
        z.sg("Bnz")

        logpx_given_z = (
            p_x_given_z.log_prob(x.unsqueeze(1).expand_as(p_x_given_z.mean))
            .sum(dim=2)
            .mean(dim=1)
            .sg("B")
        )
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1).sg("B")

        reconstruction_loss = -logpx_given_z.mean()
        kl_loss = kld.mean()
        loss = reconstruction_loss + kl_loss
        return loss, reconstruction_loss, kl_loss  # (1,)


if __name__ == "__main__":
    model = VAENCA()
    model.eval_batch()
    train(model, n_updates=100_000, eval_interval=100)
    model.test(128)
