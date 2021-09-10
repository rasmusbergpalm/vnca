import os
from typing import Sequence, Tuple

import numpy as np
import torch as t
import torch.utils.data
from PIL import Image
from shapeguard import ShapeGuard
from torch import nn, optim
from torch.distributions import Normal, Distribution, kl_divergence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard._utils import make_grid
from torchvision import transforms

from iterable_dataset_wrapper import IterableWrapper
from modules.model import Model
from modules.nca import MitosisNCA
from noto import NotoEmoji
from train import train
from util import get_writers


class DNAUpdate(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.update_net = t.nn.Sequential(
            t.nn.Conv2d(state_dim, hidden_dim, 3, padding=1),
            t.nn.ELU(),
            t.nn.Conv2d(hidden_dim, hidden_dim, 1),
            t.nn.ELU(),
            t.nn.Conv2d(hidden_dim, hidden_dim, 1),
            t.nn.ELU(),
            t.nn.Conv2d(hidden_dim, state_dim, 1, bias=False)
        )
        self.update_net[-1].weight.data.fill_(0.0)

    def forward(self, state):
        state.sg("Bzhw")
        update = self.update_net(state).sg("Bzhw")
        update[:, (self.state_dim // 2):, :, :] = 0.0  # zero out the last half (DNA)
        return update


class VAENCA(Model, nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.h = self.w = 64
        self.z_size = 128
        self.train_loss_fn = self.elbo_loss_function
        self.train_samples = 1
        self.test_loss_fn = self.iwae_loss_fn
        self.test_samples = 1
        self.hidden_size = 256

        batch_size = 16

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = nn.Sequential(
            nn.Linear(self.h * self.w * 4, self.hidden_size), nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ELU(),
            nn.Linear(self.hidden_size, 2 * self.z_size)
        )
        update_net = DNAUpdate(self.z_size, self.hidden_size)
        self.alive_channel = 3  # Alpha in RGBA
        self.nca = MitosisNCA(self.h, self.w, self.z_size, None, update_net, 5, 8, self.alive_channel, 1.0, 0.1)

        self.register_buffer("log_sigma", t.scalar_tensor(0.0, device=self.device))
        self.p_z = Normal(t.zeros(self.z_size, device=self.device), t.ones(self.z_size, device=self.device))

        data_dir = os.environ.get('DATA_DIR') or "."

        tp = transforms.Compose([transforms.Lambda(lambda img: img.convert("RGBA")), transforms.Resize((self.h, self.w)), transforms.ToTensor()])
        train_data, val_data = NotoEmoji(data_dir, tp).train_val_split()  # datasets.CelebA(data_dir, split="train", download=True, transform=tp)
        self.train_loader = iter(DataLoader(IterableWrapper(train_data), batch_size=batch_size, pin_memory=True))
        self.test_loader = iter(DataLoader(IterableWrapper(val_data), batch_size=batch_size, pin_memory=True))
        self.train_writer, self.test_writer = get_writers("hierarchical-nca")

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
        loss, z, p_x_given_z = self.forward(x, self.train_samples, self.train_loss_fn)
        loss.backward()

        for p in self.parameters():  # grad norm
            p.grad /= (t.norm(p.grad) + 1e-8)

        self.optimizer.step()

        if self.batch_idx % 100 == 0:
            self.report(self.train_writer, loss)

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
            loss, z, p_x_given_z = self.forward(x, self.test_samples, self.test_loss_fn)
            self.report(self.test_writer, loss)
        return loss.item()

    def _plot_samples(self):
        ShapeGuard.reset()
        with torch.no_grad():
            samples = self.p_z.sample((64, 1)).to(self.device)
            decode, states = self.decode(samples)
            samples = t.clip(decode.mean, 0, 1).reshape(64, 4, self.h, self.w).cpu().detach().numpy()
            samples = samples[:, :3, :, :] * samples[:, 3:4, :, :]

            growth = []
            for state in states:
                state = t.clip(state[0:1, :4, :, :], 0, 1)
                state = state[:, :3, :, :] * state[:, 3:4, :, :]  # (1, 3, h, w)
                growth.append(state)
            growth = t.cat(growth, dim=0).cpu().detach().numpy()  # (n_states, 3, h, w)

            x, y = next(self.test_loader)
            _, _, p_x_given_z = self.forward(x[:64], 1, self.iwae_loss_fn)
            recons = t.clip(p_x_given_z.mean, 0, 1).reshape(-1, 4, self.h, self.w).cpu().detach().numpy()
            recons = recons[:, :3, :, :] * recons[:, 3:4, :, :]

        return samples, recons, growth

    def plot_growth_samples(self):
        ShapeGuard.reset()
        with torch.no_grad():
            samples = self.p_z.sample((64, 1)).to(self.device)
            _, states = self.decode(samples)
            for i, state in enumerate(states):
                samples = t.clip(state[:, :4, :, :], 0, 1).cpu().detach().numpy()
                samples = samples[:, :3, :, :] * samples[:, 3:4, :, :]  # (64, 3, h, w)
                samples = (samples * 255).astype(np.uint8)
                grid = make_grid(samples).transpose(1, 2, 0)  # (HWC)
                im = Image.fromarray(grid)
                im.save("samples-%03d.png" % i)

    def report(self, writer: SummaryWriter, loss):
        writer.add_scalar('loss', loss.item(), self.batch_idx)
        writer.add_scalar('log_sigma', self.log_sigma.item(), self.batch_idx)

        samples, recons, growth = self._plot_samples()
        # writer.add_images("grid", grid, self.batch_idx)
        writer.add_images("samples", samples, self.batch_idx)
        writer.add_images("recons", recons, self.batch_idx)
        writer.add_images("growth", growth, self.batch_idx)

    def encode(self, x) -> Distribution:  # q(z|x)
        x.sg("Bx")
        q = self.encoder(x).sg("BZ")
        loc = q[:, :self.z_size].sg("Bz")
        logsigma = q[:, self.z_size:].sg("Bz")
        return Normal(loc=loc, scale=t.exp(logsigma))

    def decode(self, z: t.Tensor) -> Tuple[Distribution, Sequence[t.Tensor]]:  # p(x|z)
        z.sg("Bnz")
        bs, ns, zs = z.shape
        z[:, :, self.alive_channel] = 1.0  # Force the seed cells to be alive
        z = z.reshape((-1, self.z_size)).unsqueeze(2).unsqueeze(3).expand(-1, -1, 2, 2).sg("bz22")
        pad = [self.h // 2 - 1, self.h // 2 - 1, self.w // 2 - 1, self.w // 2 - 1]
        state = t.nn.functional.pad(z, pad, mode="constant", value=0)
        states = self.nca(state)

        outputs = states[-1][:, :4, :, :].sg("b4hw").reshape((bs, ns, -1)).sg("Bnx")

        return Normal(loc=outputs, scale=self.log_sigma.exp()), states

    def forward(self, x, n_samples, loss_fn):
        ShapeGuard.reset()
        x.sg("B4hw")
        x = x.to(self.device).reshape(x.shape[0], -1).sg("Bx")
        q_z_given_x = self.encode(x).sg("Bz")
        z = q_z_given_x.rsample((n_samples,)).permute((1, 0, 2)).sg("Bnz")
        decode, _ = self.decode(z)
        p_x_given_z = decode.sg("Bnx")

        loss = loss_fn(x, p_x_given_z, q_z_given_x, z)
        return loss, z, p_x_given_z

    def iwae_loss_fn(self, x: t.Tensor, p_x_given_z: Distribution, q_z_given_x: Distribution, z: t.Tensor) -> t.Tensor:
        """
          log(p(x)) >= logsumexp_{i=1}^N[ log(p(x|z_i)) + log(p(z_i)) - log(q(z_i|x))] - log(N)
        """
        x.sg("Bx")
        p_x_given_z.sg("Bnx")
        q_z_given_x.sg("Bz")
        z.sg("Bnz")

        logpx_given_z = p_x_given_z.log_prob(x.unsqueeze(1).expand_as(p_x_given_z.mean)).sum(dim=2).sg("Bn")
        logpz = self.p_z.log_prob(z).sum(dim=2).sg("Bn")
        logqz_given_x = q_z_given_x.log_prob(z.permute((1, 0, 2))).sum(dim=2).permute((1, 0)).sg("Bn")
        logpx = (t.logsumexp(logpx_given_z + logpz - logqz_given_x, dim=1) - t.log(t.scalar_tensor(z.shape[1]))).sg("B")
        return -logpx.mean()  # (1,)

    def elbo_loss_function(self, x: t.Tensor, p_x_given_z: Distribution, q_z_given_x: Distribution, z: t.Tensor) -> t.Tensor:
        """
          log p(x) >= E_q(z|x) [ log p(x|z) p(z) / q(z|x) ]
          Reconstruction + KL divergence losses summed over all elements and batch
        """
        x.sg("Bx")
        p_x_given_z.sg("Bnx")
        q_z_given_x.sg("Bz")
        z.sg("Bnz")

        logpx_given_z = p_x_given_z.log_prob(x.unsqueeze(1).expand_as(p_x_given_z.mean)).sum(dim=2).mean(dim=1).sg("B")
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1).sg("B")

        return (-logpx_given_z + kld).mean()  # (1,)


if __name__ == "__main__":
    model = VAENCA()
    model.eval_batch()
    train(model, n_updates=100_000, eval_interval=100)
