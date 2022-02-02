import os
from typing import Sequence, Tuple

import numpy as np
import torch as t
import torch.utils.data
import tqdm
from PIL import Image
from shapeguard import ShapeGuard
from torch import nn, optim
from torch.distributions import Normal, Distribution, kl_divergence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard._utils import make_grid
from torchvision import transforms, datasets

from iterable_dataset_wrapper import IterableWrapper
from logistic import DiscreteLogistic
from modules.dml import DiscretizedMixtureLogitsDistribution
from modules.model import Model
from modules.nca import MitosisNCA
from modules.residual import Residual
from noto import NotoEmoji
from train import train
from util import get_writers


# torch.autograd.set_detect_anomaly(True)

class VAENCA(Model, nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.h = self.w = 64
        self.z_size = 256
        self.train_loss_fn = self.elbo_loss_function
        self.train_samples = 1
        self.test_loss_fn = self.iwae_loss_fn
        self.test_samples = 1
        self.nca_hid = 256
        self.encoder_hid = 32
        self.n_mixtures = 4
        batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = "celeba"  # celeba
        assert self.dataset in {'emoji', 'celeba'}
        self.n_channels = 3

        filter_size = (5, 5)
        pad = tuple(s // 2 for s in filter_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_channels, self.encoder_hid * 2 ** 0, filter_size, padding=pad), nn.ELU(),  # (bs, 32, 64, 64)
            nn.Conv2d(self.encoder_hid * 2 ** 0, self.encoder_hid * 2 ** 1, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 64, 32, 32)
            nn.Conv2d(self.encoder_hid * 2 ** 1, self.encoder_hid * 2 ** 2, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 128, 16, 16)
            nn.Conv2d(self.encoder_hid * 2 ** 2, self.encoder_hid * 2 ** 3, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 256, 8, 8)
            nn.Conv2d(self.encoder_hid * 2 ** 3, self.encoder_hid * 2 ** 4, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 512, 4, 4),
            nn.Flatten(),  # (bs, 512*4*4)
            nn.Linear(self.encoder_hid * (2 ** 4) * 4 * 4, 2 * self.z_size),
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
            t.nn.Conv2d(self.nca_hid, self.z_size, 1)
        )
        update_net[-1].weight.data.fill_(0.0)
        update_net[-1].bias.data.fill_(0.0)

        self.nca = t.nn.DataParallel(MitosisNCA(self.h, self.w, self.z_size, update_net, 5, 8, 1.0))

        # self.log_sigma = t.nn.Parameter(-2 * t.ones((4,), device=self.device), requires_grad=True)
        self.p_z = Normal(t.zeros(self.z_size, device=self.device), t.ones(self.z_size, device=self.device))

        data_dir = os.environ.get('DATA_DIR') or "."

        tp = transforms.Compose([transforms.Lambda(lambda img: img.convert("RGB")), transforms.Resize((self.h, self.w)), transforms.ToTensor()])
        if self.dataset == 'emoji':
            train_data, val_data = NotoEmoji(data_dir, tp).train_val_split()
        elif self.dataset == 'celeba':
            train_data, val_data = datasets.CelebA(data_dir, split="train", download=True, transform=tp), datasets.CelebA(data_dir, split="valid", download=True, transform=tp)
        else:
            raise NotImplementedError()
        self.train_loader = iter(DataLoader(IterableWrapper(train_data), batch_size=batch_size, pin_memory=True))
        self.test_loader = iter(DataLoader(IterableWrapper(val_data), batch_size=batch_size, pin_memory=True))
        self.train_writer, self.test_writer = get_writers("hierarchical-nca")

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
        loss, z, p_x_given_z, recon_loss, kl_loss = self.forward(x, self.train_samples, self.train_loss_fn)
        loss.backward()

        t.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        self.optimizer.step()

        if self.batch_idx % 100 == 0:
            self.report(self.train_writer, p_x_given_z, loss, recon_loss, kl_loss)

        self.batch_idx += 1
        return loss.item()

    def save(self, fn):
        t.save({
            'batch_idx': self.batch_idx,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, fn)

    def load(self, fn):
        checkpoint = t.load(fn, map_location=t.device(self.device))
        self.batch_idx = checkpoint["batch_idx"]
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def eval_batch(self):
        self.train(False)
        with t.no_grad():
            x, y = next(self.test_loader)
            loss, z, p_x_given_z, recon_loss, kl_loss = self.forward(x, self.test_samples, self.test_loss_fn)
            self.report(self.test_writer, p_x_given_z, loss, recon_loss, kl_loss)
        return loss.item()

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
            recons = self.to_rgb(p_x_given_z.logits)

        return samples, recons, growth

    def to_rgb(self, state):
        samples = []
        dml = DiscretizedMixtureLogitsDistribution(self.n_mixtures, state[:, :self.n_mixtures * 10, :, :])
        for _ in range(1000):
            samples.append((dml.sample() + 1) / 2)

        return t.stack(samples, dim=0).mean(dim=0)

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
        writer.add_scalar('loss', loss.item(), self.batch_idx)
        writer.add_scalar('bpd', loss.item() / (np.log(2) * self.h * self.w * self.n_channels), self.batch_idx)
        # writer.add_scalar('log_sigma', p_x_given_z.logscale.mean().item(), self.batch_idx)
        if recon_loss:
            writer.add_scalar('recon_loss', recon_loss.item(), self.batch_idx)
        if kl_loss:
            writer.add_scalar('kl_loss', kl_loss.item(), self.batch_idx)

        samples, recons, growth = self._plot_samples()
        # writer.add_images("grid", grid, self.batch_idx)
        writer.add_images("samples", samples, self.batch_idx)
        writer.add_images("recons", recons, self.batch_idx)
        writer.add_images("growth", growth, self.batch_idx)

    def encode(self, x) -> Distribution:  # q(z|x)
        x.sg("*4hw")
        q = self.encoder(x).sg("*Z")
        loc = t.clip(q[:, :self.z_size].sg("*z"), -100.0, 100.0)
        logsigma = t.clip(q[:, self.z_size:].sg("*z"), -7.0, 7.0)
        return Normal(loc=loc, scale=t.exp(logsigma))

    def decode(self, z: t.Tensor) -> Tuple[Distribution, Sequence[t.Tensor]]:  # p(x|z)
        z.sg("*nz")
        bs, ns, zs = z.shape
        state = z.reshape((-1, self.z_size)).unsqueeze(2).unsqueeze(3).expand(-1, -1, 2, 2).sg("*z22")
        states = self.nca(state)

        # states = [self.decoder(state) for state in states]

        state = states[-1]

        logits = state[:, :self.n_mixtures * 10, :, :]
        # logscale = t.zeros_like(loc)
        # logscale = self.log_sigma.unsqueeze(0).unsqueeze(2).unsqueeze(3).sg((1, 4, 1, 1)).expand_as(state[:, :4, :, :]).reshape((bs, ns, -1)).sg("*nx")

        return DiscretizedMixtureLogitsDistribution(self.n_mixtures, logits), states

    def forward(self, x, n_samples, loss_fn):
        ShapeGuard.reset()
        x.sg(("B", 3, "h", "w"))
        x = (x.to(self.device) * 2) - 1
        q_z_given_x = self.encode(x).sg("*z")
        z = q_z_given_x.rsample((n_samples,)).permute((1, 0, 2)).sg("*nz")
        decode, _ = self.decode(z)
        p_x_given_z = decode.sg("**hw")

        loss, recon_loss, kl_loss = loss_fn(x, p_x_given_z, q_z_given_x, z)
        return loss, z, p_x_given_z, recon_loss, kl_loss

    def iwae_loss_fn(self, x: t.Tensor, p_x_given_z: Distribution, q_z_given_x: Distribution, z: t.Tensor):
        """
          log(p(x)) >= logsumexp_{i=1}^N[ log(p(x|z_i)) + log(p(z_i)) - log(q(z_i|x))] - log(N)
        """
        x.sg(("B", 3, "h", "w"))
        p_x_given_z.sg(("b", self.n_mixtures * 10, "h", "w"))
        q_z_given_x.sg("*z")
        z.sg("*nz")
        B, n, zs = z.shape

        x = (x.unsqueeze(1)
             .expand((-1, n, -1, -1, -1))
             .reshape(-1, 3, self.h, self.w)
             ).sg(("b", 3, self.h, self.w))
        logpx_given_z = p_x_given_z.log_prob(x).sum(dim=(1, 2)).sg("*").reshape((B, n))
        logpz = self.p_z.log_prob(z).sum(dim=2).sg("*n")
        logqz_given_x = q_z_given_x.log_prob(z.permute((1, 0, 2))).sum(dim=2).permute((1, 0)).sg("*n")
        logpx = (t.logsumexp(logpx_given_z + logpz - logqz_given_x, dim=1) - t.log(t.scalar_tensor(z.shape[1]))).sg("*")
        return -logpx.mean(), None, None  # (1,)

    def elbo_loss_function(self, x: t.Tensor, p_x_given_z: Distribution, q_z_given_x: Distribution, z: t.Tensor):
        """
          log p(x) >= E_q(z|x) [ log p(x|z) p(z) / q(z|x) ]
          Reconstruction + KL divergence losses summed over all elements and batch
        """
        x.sg(("B", 3, "h", "w"))
        p_x_given_z.sg(("b", self.n_mixtures * 10, "h", "w"))
        q_z_given_x.sg("*z")
        z.sg("*nz")
        B, n, zs = z.shape

        x = (x.unsqueeze(1)
             .expand((-1, n, -1, -1, -1))
             .reshape(-1, 3, self.h, self.w)
             ).sg(("b", 3, self.h, self.w))
        logpx_given_z = p_x_given_z.log_prob(x).sum(dim=(1, 2)).sg("*").reshape((B, n)).mean(dim=1).sg("*")
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1).sg("*")

        reconstruction_loss = -logpx_given_z.mean()
        kl_loss = kld.mean()

        loss = reconstruction_loss + 100 * kl_loss
        return loss, reconstruction_loss, kl_loss  # (1,)

    def twitter_gif(self):
        with t.no_grad():
            samples = self.p_z.sample((64,)).view(64, 1, -1).to(self.device)
            dml, states = self.decode(samples)  # n,bchw

            means = [self.to_rgb(state) for state in tqdm.tqdm(states)]

        t.save({'means': means}, 'means.t')


if __name__ == "__main__":
    model = VAENCA()
    model.load('../8d1ee6e/latest')
    model.twitter_gif()
