import os

import matplotlib.pyplot as plt
import torch as t
import torch.utils.data
import tqdm as tqdm
from shapeguard import ShapeGuard
from torch import nn, optim
from torch.distributions import Normal, Distribution, Binomial, kl_divergence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from iterable_dataset_wrapper import IterableWrapper
from modules.nca import MitosisNCA
from util import get_writers


class DNAUpdate(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.update_net = t.nn.Sequential(
            t.nn.Conv2d(state_dim, 128, 3, padding=1),
            t.nn.Tanh(),
            t.nn.Conv2d(128, 128, 1),
            t.nn.Tanh(),
            t.nn.Conv2d(128, state_dim, 1, bias=False)
        )
        self.update_net[-1].weight.data.fill_(0.0)

    def forward(self, state):
        state.sg("Bzhw")
        update = self.update_net(state).sg("Bzhw")
        update[:, (self.state_dim // 2):, :, :] = 0.0  # zero out the last half (DNA)
        return update


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.h = self.w = 32
        self.z_size = 32
        self.train_loss_fn = self.elbo_loss_function
        self.train_samples = 1
        self.test_loss_fn = self.iwae_loss_fn
        self.test_samples = 1
        self.hidden_size = 512

        batch_size = 128

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = nn.Sequential(
            nn.Linear(32 ** 2, self.hidden_size), nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ELU(),
            nn.Linear(self.hidden_size, 2 * self.z_size)
        )
        update_net = DNAUpdate(self.z_size)
        self.nca = MitosisNCA(self.h, self.w, self.z_size, update_net, 4, 8, 0, 1.0, 0.1)
        self.out_w = t.nn.Parameter(t.scalar_tensor(1.0), requires_grad=True)
        self.out_b = t.nn.Parameter(t.scalar_tensor(-5.0), requires_grad=True)
        """
        self.decoder = nn.Sequential(
            nn.Linear(self.z_size, self.hidden_size), nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ELU(),
            nn.Linear(self.hidden_size, 32 ** 2)
        )
        """

        self.p_z = Normal(t.zeros(self.z_size, device=self.device), t.ones(self.z_size, device=self.device))

        data_dir = os.environ.get('DATA_DIR') or "."
        tp = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])  # 32, 32
        self.train_loader = iter(DataLoader(IterableWrapper(datasets.MNIST(data_dir, train=True, download=True, transform=tp)), batch_size=batch_size, pin_memory=True))
        self.test_loader = iter(DataLoader(IterableWrapper(datasets.MNIST(data_dir, train=False, transform=tp)), batch_size=batch_size, pin_memory=True))
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

    def test_batch(self):
        self.train(False)
        with t.no_grad():
            x, y = next(self.test_loader)
            loss, z, p_x_given_z = self.forward(x, self.test_samples, self.test_loss_fn)
            self.report(self.test_writer, loss)

    def _plot_samples(self):
        ShapeGuard.reset()
        with torch.no_grad():
            # grid = t.stack([t.tensor([x, y]) for x in torch.linspace(-3, 3, 8) for y in torch.linspace(-3, 3, 8)]).reshape(64, 1, 2).to(self.device)
            # grid = self.decode(grid).sample().reshape(64, 1, self.h, self.w).cpu().detach().numpy()

            samples = self.p_z.sample((64, 1)).to(self.device)
            samples = self.decode(samples).sample().reshape(64, 1, self.h, self.w).cpu().detach().numpy()

            x, y = next(self.test_loader)
            _, _, p_x_given_z = self.forward(x[:64], 1, self.iwae_loss_fn)
            recons = p_x_given_z.sample().reshape(64, 1, self.h, self.w).cpu().detach().numpy()

        return samples, recons

    def report(self, writer: SummaryWriter, loss):
        writer.add_scalar('loss', loss.item(), self.batch_idx)

        samples, recons = self._plot_samples()
        # writer.add_images("grid", grid, self.batch_idx)
        writer.add_images("samples", samples, self.batch_idx)
        writer.add_images("recons", recons, self.batch_idx)

    def encode(self, x) -> Distribution:  # q(z|x)
        x.sg("Bx")
        q = self.encoder(x).sg("BZ")
        loc = q[:, :self.z_size].sg("Bz")
        logsigma = q[:, self.z_size:].sg("Bz")
        return Normal(loc=loc, scale=t.exp(logsigma))

    def decode_vae(self, z: t.Tensor) -> Distribution:
        z.sg("Bnz")
        return Binomial(1, logits=self.decoder(z)).sg("Bnx")

    def decode(self, z: t.Tensor) -> Distribution:  # p(x|z)
        z.sg("Bnz")
        bs, ns, zs = z.shape
        z[:, :, 0] = 1.0  # Force the seed cells to be alive
        z = z.reshape((-1, self.z_size)).unsqueeze(2).unsqueeze(3).expand(-1, -1, 2, 2).sg("bz22")
        state = t.nn.functional.pad(z, [15, 15, 15, 15], mode="constant", value=0)
        states = self.nca(state)
        state = states[-1].sg("bzhw").reshape((bs, ns, self.z_size, -1)).sg("Bnzx")
        # probs = t.clip(state[:, :, 0, :], 0.0, 1.0)
        logits = state[:, :, 0, :] * self.out_w + self.out_b
        return Binomial(1, logits=logits).sg("Bnx")

    def forward(self, x, n_samples, loss_fn):
        ShapeGuard.reset()
        x.sg(("B", 1, "h", "w"))
        x = x.to(self.device).reshape(x.shape[0], -1).sg("Bx")
        x = Binomial(probs=x).sample()
        q_z_given_x = self.encode(x).sg("Bz")
        z = q_z_given_x.rsample((n_samples,)).permute((1, 0, 2)).sg("Bnz")
        p_x_given_z = self.decode(z).sg("Bnx")
        loss = loss_fn(x, p_x_given_z, q_z_given_x, z)
        return loss, z, p_x_given_z

    def iwae_loss_fn(self, x: t.Tensor, p_x_given_z: Distribution, q_z_given_x: Distribution, z: t.Tensor) -> t.Tensor:
        """
          log(p(x)) >= logsumexp_{i=1}^N[ log(p(x|z_i)) + log(p(z_i)) - log(q(z_i|x))] - log(N)
          x: (bs, 784)
          p_x_given_z: (bs, n_samples, 784)
          q_z_given_x: (bs, z_size)
          z: (bs, n_samples, z_size)
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
          x: (bs, h, w)
          p_x_given_z: (bs, n_samples, hw)
          q_z_given_x: (bs, z_size)
          z: (bs, n_samples, z_size)
        """

        logpx_given_z = p_x_given_z.log_prob(x.unsqueeze(1).expand_as(p_x_given_z.mean)).sum(dim=2).mean(dim=1).sg("B")
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1).sg("B")

        return (-logpx_given_z + kld).mean()  # (1,)


def agg_post(mus, ys):
    mus = t.cat(mus).cpu().numpy()
    ys = t.cat(ys).cpu().numpy()
    cmap = plt.get_cmap("tab10")
    colors = [cmap(y) for y in ys]
    plt.figure(figsize=(8, 8))
    plt.scatter(mus[:, 0], mus[:, 1], alpha=0.25, c=colors, marker='.')
    plt.close()


if __name__ == "__main__":
    # Distribution.set_default_validate_args(True)
    n_train_updates = 100_000
    n_test_batches = 100

    model = Model()
    model.test_batch()
    for _ in tqdm.tqdm(range(n_train_updates)):
        model.train_batch()
        if model.batch_idx % n_test_batches == 0:
            model.test_batch()
