import torch as t
from torch.distributions import Distribution, kl_divergence


def iwae(x: t.Tensor, p_x_given_z: Distribution, q_z_given_x: Distribution, p_z: Distribution, z: t.Tensor):
    """
        log(p(x)) >= logsumexp_{i=1}^N[ log(p(x|z_i)) + log(p(z_i)) - log(q(z_i|x))] - log(N)

        x: bchw
        q_z_given_x: bz
        z: bnz
        p_x_given_z: (bn)chw
    """
    b, c, h, w = x.shape
    b, n, zs = z.shape

    x = (x.unsqueeze(1)
         .expand((-1, n, -1, -1, -1))
         .reshape(b * n, c, h, w)
         )  # (bn)chw
    logpx_given_z = p_x_given_z.log_prob(x).sum(dim=(1, 2, 3)).reshape((b, n))
    logpz = p_z.log_prob(z).sum(dim=2)
    logqz_given_x = q_z_given_x.log_prob(z.permute((1, 0, 2))).sum(dim=2).permute((1, 0))
    logpx = (t.logsumexp(logpx_given_z + logpz - logqz_given_x, dim=1) - t.log(t.scalar_tensor(z.shape[1])))

    return -logpx, None, None


def elbo(x: t.Tensor, p_x_given_z: Distribution, q_z_given_x: Distribution, p_z: Distribution, z: t.Tensor):
    """
        log p(x) >= E_q(z|x) [ log p(x|z) p(z) / q(z|x) ]
        Reconstruction + KL divergence losses summed over all elements and batch

        x: bchw
        q_z_given_x: bz
        z: bnz
        p_x_given_z: (bn)chw
    """

    b, c, h, w = x.shape
    b, n, zs = z.shape

    x = (x.unsqueeze(1)
         .expand((-1, n, -1, -1, -1))
         .reshape(-1, c, h, w)
         )  # (bn)chw

    logpx_given_z = p_x_given_z.log_prob(x).sum(dim=(1, 2, 3)).reshape((b, n)).mean(dim=1)
    kld = kl_divergence(q_z_given_x, p_z).sum(dim=1)

    reconstruction_loss = -logpx_given_z
    kl_loss = kld

    loss = reconstruction_loss + kl_loss
    return loss, reconstruction_loss, kl_loss
