import os

import matplotlib.pyplot as plt
import torch as t
import torch.utils.data
import tqdm as tqdm
from torch import nn, optim
from torch.distributions import Normal, Distribution, Binomial, kl_divergence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from iterable_dataset_wrapper import IterableWrapper
from util import get_writers
from shapeguard import ShapeGuard


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.z_size = 16
        self.train_samples = 1
        self.train_loss_fn = self.elbo_loss_function
        self.test_loss_fn = self.iwae_loss_fn

        batch_size = 256
        self.hidden_size = 512
        self.test_samples = 1024
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = nn.Sequential(
            nn.Linear(784, self.hidden_size), nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ELU(),
            nn.Linear(self.hidden_size, 2 * self.z_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_size, self.hidden_size), nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size), nn.ELU(),
            nn.Linear(self.hidden_size, 784)
        )

        self.p_z = Normal(t.zeros(self.z_size, device=self.device), t.ones(self.z_size, device=self.device))

        data_dir = os.environ.get('DATA_DIR') or "."
        self.train_loader = iter(DataLoader(IterableWrapper(datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())), batch_size=batch_size, pin_memory=True))
        self.test_loader = iter(DataLoader(IterableWrapper(datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor())), batch_size=batch_size, pin_memory=True))
        self.train_writer, self.test_writer = get_writers("mnist")

        print(self)
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)
        self.batch_idx = 0

    def train_batch(self):
        x, y = next(self.train_loader)
        self.train(True)
        self.optimizer.zero_grad()
        loss, z, p_x_given_z = self.forward(x, self.train_samples, self.train_loss_fn)
        loss.backward()
        self.optimizer.step()

        if self.batch_idx % 100 == 0:
            self.report(self.train_writer, loss)

        self.batch_idx += 1

    def test_batch(self):
        x, y = next(self.test_loader)
        self.train(False)
        with t.no_grad():
            loss, z, p_x_given_z = self.forward(x, self.test_samples, self.test_loss_fn)
            self.report(self.test_writer, loss)

    def _plot_samples(self):
        ShapeGuard.reset()
        with torch.no_grad():
            # grid = t.stack([t.tensor([x, y]) for x in torch.linspace(-3, 3, 8) for y in torch.linspace(-3, 3, 8)]).view(64, 1, 2).to(self.device)
            # grid = self.decode(grid).mean.view(64, 1, 28, 28).cpu().detach().numpy()

            samples = self.p_z.sample((64, 1)).to(self.device)
            samples = self.decode(samples).mean.view(64, 1, 28, 28).cpu().detach().numpy()

        return samples

    def report(self, writer: SummaryWriter, loss):
        writer.add_scalar('loss', loss.item(), self.batch_idx)

        samples = self._plot_samples()
        # writer.add_images("grid", grid, self.batch_idx)
        writer.add_images("samples", samples, self.batch_idx)

    def encode(self, x) -> Distribution:  # q(z|x)
        x.sg("bx")
        q = self.encoder(x).sg("bZ")
        loc = q[:, :self.z_size].sg("bz")
        logsigma = q[:, self.z_size:].sg("bz")
        return Normal(loc=loc, scale=t.exp(logsigma))

    def decode(self, z) -> Distribution:  # p(x|z)
        z.sg("bnz")
        decoder = self.decoder(z).sg("bnx")
        return Binomial(1, logits=decoder)

    def forward(self, x, n_samples, loss_fn):
        ShapeGuard.reset()
        x.sg(("b", 1, "h", "w"))
        """
        :param x: (bs, 1, 28, 28)
        :return:
        """
        x = x.to(self.device).view(x.shape[0], -1).sg("bx")
        x = Binomial(probs=x).sample()
        q_z_given_x = self.encode(x).sg("bz")
        z = q_z_given_x.rsample((n_samples,)).permute((1, 0, 2)).sg("bnz")
        p_x_given_z = self.decode(z).sg("bnx")
        loss = loss_fn(x, p_x_given_z, q_z_given_x, z)
        return loss, z, p_x_given_z

    def iwae_loss_fn(self, x: t.Tensor, p_x_given_z: Distribution, q_z_given_x: Distribution, z: t.Tensor) -> t.Tensor:
        """
          log(p(x)) \approx logsumexp_{i=1}^N[ log(p(x|z_i)) + log(p(z_i)) - log(q(z_i|x))] - log(N)
          x: (bs, 784)
          p_x_given_z: (bs, n_samples, 784)
          q_z_given_x: (bs, z_size)
          z: (bs, n_samples, z_size)
        """
        x.sg("bx")
        p_x_given_z.sg("bnx")
        q_z_given_x.sg("bz")
        z.sg("bnz")

        logpx_given_z = p_x_given_z.log_prob(x.unsqueeze(1).expand_as(p_x_given_z.mean)).sum(dim=2).sg("bn")
        logpz = self.p_z.log_prob(z).sum(dim=2).sg("bn")
        logqz_given_x = q_z_given_x.log_prob(z.permute((1, 0, 2))).sum(dim=2).permute((1, 0)).sg("bn")
        logpx = (t.logsumexp(logpx_given_z + logpz - logqz_given_x, dim=1) - t.log(t.scalar_tensor(z.shape[1]))).sg("b")
        return -logpx.mean()  # (1,)

    def elbo_loss_function(self, x: t.Tensor, p_x_given_z: Distribution, q_z_given_x: Distribution, z: t.Tensor) -> t.Tensor:
        """
          log p(x) >= E_q(z|x) [ log p(x|z) p(z) / q(z|x) ]
          Reconstruction + KL divergence losses summed over all elements and batch
          x: (bs, 28, 28)
          p_x_given_z: (bs, n_samples, 784)
          q_z_given_x: (bs, z_size)
          z: (bs, n_samples, z_size)
        """

        logpx_given_z = p_x_given_z.log_prob(x.unsqueeze(1).expand_as(p_x_given_z.mean)).sum(dim=2).mean(dim=1)  # (bs, )
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1)  # (bs, )

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
