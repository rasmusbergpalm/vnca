import os
import random
from typing import Sequence, Tuple

import numpy as np
import torch as t
import torch.utils.data
import tqdm
from shapeguard import ShapeGuard
from torch import nn, optim
from torch.distributions import Normal, Distribution, kl_divergence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from modules.dml import DiscretizedMixtureLogitsDistribution
from modules.iterable_dataset_wrapper import IterableWrapper
from modules.loss import elbo, iwae
from modules.model import Model
from modules.nca import NCA
from train import train
from util import get_writers


# torch.autograd.set_detect_anomaly(True)

class VNCA(Model):
    def __init__(self):
        super(Model, self).__init__()
        self.h = self.w = 64
        self.z_size = 15
        self.train_loss_fn = elbo
        self.train_samples = 1
        self.test_loss_fn = iwae
        self.test_samples = 3
        self.nca_hid = 7
        self.encoder_hid = 32
        self.n_mixtures = 1
        self.bpd_dimensions = 3 * 64 * 64
        batch_size = 32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_channels = 3
        self.pool = []
        self.pool_size = 1024
        self.n_damage = 8
        self.dmg_size = 14

        filter_size = (5, 5)
        pad = tuple(s // 2 for s in filter_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_channels, self.encoder_hid * 2 ** 0, filter_size, padding=pad), nn.ELU(),  # (bs, 32, h, w)
            nn.Conv2d(self.encoder_hid * 2 ** 0, self.encoder_hid * 2 ** 1, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 64, h//2, w//2)
            nn.Conv2d(self.encoder_hid * 2 ** 1, self.encoder_hid * 2 ** 2, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 128, h//4, w//4)
            nn.Conv2d(self.encoder_hid * 2 ** 2, self.encoder_hid * 2 ** 3, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 256, h//8, w//8)
            nn.Conv2d(self.encoder_hid * 2 ** 3, self.encoder_hid * 2 ** 4, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 512, h//16, w//16),
            nn.Flatten(),  # (bs, 512*h//16*w//16)
            nn.Linear(self.encoder_hid * (2 ** 4) * self.h // 16 * self.w // 16, 2 * self.z_size),
        )

        update_net = t.nn.Sequential(
            t.nn.Conv2d(self.z_size, self.nca_hid, 3, padding=1),
            t.nn.ELU(),
            t.nn.Conv2d(self.nca_hid, self.z_size, 1, bias=False)
        )
        update_net[-1].weight.data.fill_(0.0)
        # update_net[-1].bias.data.fill_(0.0)

        self.nca = NCA(update_net, 32, 64, 0.5)
        self.p_z = Normal(t.zeros(self.z_size, device=self.device), t.ones(self.z_size, device=self.device))

        data_dir = os.environ.get('DATA_DIR') or "data"

        tp = transforms.Compose([transforms.Resize((self.h, self.w)), transforms.ToTensor()])
        train_data, val_data = datasets.CelebA(data_dir, split="train", download=True, transform=tp), datasets.CelebA(data_dir, split="valid", download=True, transform=tp)

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
        loss, z, p_x_given_z, recon_loss, kl_loss, states = self.forward(x, self.train_samples, self.train_loss_fn)
        loss.mean().backward()

        t.nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)

        self.optimizer.step()

        if self.batch_idx % 100 == 0:
            self.report(self.train_writer, p_x_given_z, loss, recon_loss, kl_loss)

        self.batch_idx += 1
        return loss.mean().item()

    def eval_batch(self):
        self.train(False)
        with t.no_grad():
            x, y = next(self.test_loader)
            loss, z, p_x_given_z, recon_loss, kl_loss, states = self.forward(x, self.test_samples, self.test_loss_fn)
            self.report(self.test_writer, p_x_given_z, loss, recon_loss, kl_loss)
        return loss.mean().item()

    def test(self, n_iw_samples):
        self.train(False)
        with t.no_grad():
            total_loss = 0.0
            for x, y in tqdm.tqdm(self.test_set.samples):
                loss, z, p_x_given_z, recon_loss, kl_loss, states = self.forward(x, n_iw_samples, self.test_loss_fn)
                total_loss += loss.mean().item()

        print(total_loss / len(self.test_set))

    def _plot_samples(self, writer):
        ShapeGuard.reset()
        with torch.no_grad():
            # samples
            samples = self.p_z.sample((64,)).view(64, -1, 1, 1).expand(64, -1, self.h, self.w).to(self.device)
            states = self.decode(samples)
            samples, samples_means = self.to_rgb(states[-1])
            writer.add_images("samples/samples", samples, self.batch_idx)
            writer.add_images("samples/means", samples_means, self.batch_idx)

            # Growths
            growth_samples = []
            growth_means = []
            for state in states:
                growth_sample, growth_mean = self.to_rgb(state[0:1])
                growth_samples.append(growth_sample)
                growth_means.append(growth_mean)

            growth_samples = t.cat(growth_samples, dim=0).cpu().detach().numpy()  # (n_states, 3, h, w)
            growth_means = t.cat(growth_means, dim=0).cpu().detach().numpy()  # (n_states, 3, h, w)
            writer.add_images("growth/samples", growth_samples, self.batch_idx)
            writer.add_images("growth/means", growth_means, self.batch_idx)

            # Damage
            state = states[-1]
            _, original_means = self.to_rgb(state)
            writer.add_images("dmg/1-pre", original_means, self.batch_idx)
            dmg = self.damage(state)
            _, dmg_means = self.to_rgb(dmg)
            writer.add_images("dmg/2-dmg", dmg_means, self.batch_idx)
            recovered = self.nca(state)
            _, recovered_means = self.to_rgb(recovered[-1])
            writer.add_images("dmg/3-post", recovered_means, self.batch_idx)

            # Reconstructions
            x, y = next(self.test_loader)
            _, _, p_x_given_z, _, _, states = self.forward(x[:64], 1, self.test_loss_fn)
            recons_samples, recons_means = self.to_rgb(states[-1])
            writer.add_images("recons/samples", recons_samples, self.batch_idx)
            writer.add_images("recons/means", recons_means, self.batch_idx)

            # Pool
            if len(self.pool) > 0:
                pool_xs, pool_states, pool_losses = zip(*random.sample(self.pool, min(len(self.pool), 64)))
                pool_states = t.stack(pool_states)  # 64, z, h, w
                pool_samples, pool_means = self.to_rgb(pool_states)
                writer.add_images("pool/samples", pool_samples, self.batch_idx)
                writer.add_images("pool/means", pool_means, self.batch_idx)

    def to_rgb(self, state):
        dml = DiscretizedMixtureLogitsDistribution(self.n_mixtures, state[:, :self.n_mixtures * 10, :, :])
        return (dml.sample() + 1) / 2, (dml.sample() + 1) / 2  # todo mean

    def report(self, writer: SummaryWriter, p_x_given_z, loss, recon_loss, kl_loss):
        writer.add_scalar('loss', loss.mean().item(), self.batch_idx)
        writer.add_scalar('bpd', loss.mean().item() / (np.log(2) * self.bpd_dimensions), self.batch_idx)
        writer.add_scalar('pool_size', len(self.pool), self.batch_idx)
        if recon_loss is not None:
            writer.add_scalar('recon_loss', recon_loss.mean().item(), self.batch_idx)
        if kl_loss is not None:
            writer.add_scalar('kl_loss', kl_loss.mean().item(), self.batch_idx)

        self._plot_samples(writer)

    def encode(self, x) -> Distribution:  # q(z|x)
        x.sg(("B", 3, "h", "w"))
        q = self.encoder(x).sg("BZ")
        loc = q[:, :self.z_size].sg("Bz")
        logsigma = q[:, self.z_size:].sg("Bz")
        return Normal(loc=loc, scale=t.exp(logsigma))

    def decode(self, z: t.Tensor) -> Tuple[Distribution, Sequence[t.Tensor]]:  # p(x|z)
        z.sg("bzhw")
        return self.nca(z)

    def damage(self, states):
        states.sg("*zhw")
        mask = t.ones_like(states)
        for i in range(states.shape[0]):
            h1 = random.randint(0, states.shape[2] - self.dmg_size)
            w1 = random.randint(0, states.shape[3] - self.dmg_size)
            mask[i, :, h1:h1 + self.dmg_size, w1:w1 + self.dmg_size] = 0.0
        return states * mask

    def forward(self, x, n_samples, loss_fn):
        ShapeGuard.reset()
        x.sg(("B", 3, "h", "w"))
        x = x.to(self.device)

        # Pool samples
        bs = x.shape[0]
        n_pool_samples = bs // 2
        pool_states = None
        if self.training and len(self.pool) > n_pool_samples:
            # pop n_pool_samples worst in the pool
            pool_samples = self.pool[:n_pool_samples]
            self.pool = self.pool[n_pool_samples:]

            pool_x, pool_states, _ = zip(*pool_samples)
            pool_x = t.stack(pool_x).to(self.device)
            pool_states = t.stack(pool_states).to(self.device)
            pool_states[:self.n_damage] = self.damage(pool_states[:self.n_damage])
            x[-n_pool_samples:] = pool_x

        q_z_given_x = self.encode(x).sg("Bz")
        z = q_z_given_x.rsample((n_samples,)).permute((1, 0, 2)).sg("Bnz")

        seeds = (z.reshape((-1, self.z_size))  # stuff samples into batch dimension
                 .unsqueeze(2)
                 .unsqueeze(3)
                 .expand(-1, -1, self.h, self.w).sg("bzhw"))

        if pool_states is not None:
            seeds = seeds.clone()
            seeds[-n_pool_samples:] = pool_states  # yes this is wrong and will mess up the gradient.

        states = self.decode(seeds)

        p_x_given_z = DiscretizedMixtureLogitsDistribution(self.n_mixtures, states[-1][:, :self.n_mixtures * 10, :, :])
        loss, recon_loss, kl_loss = loss_fn(x, p_x_given_z, q_z_given_x, self.p_z, z)

        if self.training:
            # Add states to pool
            def split(tensor: t.Tensor):
                return [x for x in tensor]

            self.pool += list(zip(split(x.cpu()), split(states[-1].detach().cpu()), loss.tolist()))
            # Retain the worst
            # self.pool = sorted(self.pool, key=lambda x: x[-1], reverse=True)
            random.shuffle(self.pool)
            self.pool = self.pool[:self.pool_size]

        return loss, z, p_x_given_z, recon_loss, kl_loss, states


if __name__ == "__main__":
    model = VNCA()
    model.eval_batch()
    train(model, n_updates=100_000, eval_interval=100)
    model.test(128)
