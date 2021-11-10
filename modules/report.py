import random

import torch as t
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from shapeguard import ShapeGuard


def report(model, writer: SummaryWriter, p_x_given_z, loss, recon_loss, kl_loss):
    writer.add_scalar('loss', loss.mean().item(), model.batch_idx)
    writer.add_scalar('bpd', loss.mean().item() / (np.log(2) * model.bpd_dimensions), model.batch_idx)
    writer.add_scalar('pool_size', len(model.pool), model.batch_idx)

    if recon_loss is not None:
        writer.add_scalar('recon_loss', recon_loss.mean().item(), model.batch_idx)
    if kl_loss is not None:
        writer.add_scalar('kl_loss', kl_loss.mean().item(), model.batch_idx)

    ShapeGuard.reset()
    with t.no_grad():
        # samples
        samples = model.p_z.sample((64,)).view(64, -1, 1, 1).expand(64, -1, model.h, model.w).to(model.device)
        states = model.decode(samples)
        samples, samples_means = model.to_rgb(states[-1])
        writer.add_images("samples/samples", samples, model.batch_idx)
        writer.add_images("samples/means", samples_means, model.batch_idx)

        # Growths
        growth_samples = []
        growth_means = []
        for state in states:
            growth_sample, growth_mean = model.to_rgb(state[0:1])
            growth_samples.append(growth_sample)
            growth_means.append(growth_mean)

        growth_samples = t.cat(growth_samples, dim=0).cpu().detach().numpy()  # (n_states, 3, h, w)
        growth_means = t.cat(growth_means, dim=0).cpu().detach().numpy()  # (n_states, 3, h, w)
        writer.add_images("growth/samples", growth_samples, model.batch_idx)
        writer.add_images("growth/means", growth_means, model.batch_idx)

        # Damage
        state = states[-1]
        _, original_means = model.to_rgb(state)
        writer.add_images("dmg/1-pre", original_means, model.batch_idx)
        dmg = model.damage(state)
        _, dmg_means = model.to_rgb(dmg)
        writer.add_images("dmg/2-dmg", dmg_means, model.batch_idx)
        recovered = model.nca(state)
        _, recovered_means = model.to_rgb(recovered[-1])
        writer.add_images("dmg/3-post", recovered_means, model.batch_idx)

        # Reconstructions
        x, y = next(model.test_loader)
        _, _, p_x_given_z, _, _, states = model.forward(x[:64], 1, model.test_loss_fn)
        recons_samples, recons_means = model.to_rgb(states[-1])
        writer.add_images("recons/samples", recons_samples, model.batch_idx)
        writer.add_images("recons/means", recons_means, model.batch_idx)

        # Pool
        if len(model.pool) > 0:
            pool_xs, pool_states, pool_losses = zip(*random.sample(model.pool, min(len(model.pool), 64)))
            pool_states = t.stack(pool_states)  # 64, z, h, w
            pool_samples, pool_means = model.to_rgb(pool_states)
            writer.add_images("pool/samples", pool_samples, model.batch_idx)
            writer.add_images("pool/means", pool_means, model.batch_idx)
