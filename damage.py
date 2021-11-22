import os

import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchvision import transforms, datasets
from PIL import Image

from modules.vae import VAE


def load_model() -> VAE:
    z_size = 256
    nca_hid = 128
    n_mixtures = 1
    batch_size = 32
    dmg_size = 16
    p_update = 1.0
    min_steps, max_steps = 64, 128

    filter_size = 5
    pad = filter_size // 2
    encoder_hid = 32
    h = w = 64
    n_channels = 3

    encoder = nn.Sequential(
        nn.Conv2d(n_channels, encoder_hid * 2 ** 0, filter_size, padding=pad),
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
        nn.Linear(encoder_hid * (2 ** 4) * h // 16 * w // 16, 2 * z_size),
    )

    decoder_linear = nn.Sequential(
        nn.Linear(z_size, encoder_hid * (2 ** 5) * h // 32 * w // 32), nn.ELU()
    )

    decoder = nn.Sequential(
        nn.ConvTranspose2d(
            encoder_hid * 2 ** 5,
            encoder_hid * 2 ** 4,
            filter_size,
            padding=pad,
            stride=2,
            output_padding=1,
        ),
        nn.ELU(),
        nn.ConvTranspose2d(
            encoder_hid * 2 ** 4,
            encoder_hid * 2 ** 3,
            filter_size,
            padding=pad,
            stride=2,
            output_padding=1,
        ),
        nn.ELU(),
        nn.ConvTranspose2d(
            encoder_hid * 2 ** 3,
            encoder_hid * 2 ** 2,
            filter_size,
            padding=pad,
            stride=2,
            output_padding=1,
        ),
        nn.ELU(),
        nn.ConvTranspose2d(
            encoder_hid * 2 ** 2,
            encoder_hid * 2 ** 1,
            filter_size,
            padding=pad,
            stride=2,
            output_padding=1,
        ),
        nn.ELU(),
        nn.ConvTranspose2d(
            encoder_hid * 2 ** 1,
            encoder_hid * 2 ** 0,
            filter_size,
            padding=pad,
            stride=2,
            output_padding=1,
        ),
        nn.ELU(),
        nn.ConvTranspose2d(
            encoder_hid * 2 ** 0, n_mixtures * 10, filter_size, padding=pad
        ),
    )

    # encoder = DataParallel(encoder)
    # update_net = DataParallel(update_net)

    data_dir = os.environ.get("DATA_DIR") or "data"
    tp = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])
    train_data, val_data, test_data = [
        datasets.CelebA(data_dir, split=split, download=True, transform=tp)
        for split in ["train", "valid", "test"]
    ]

    vae = VAE(
        h,
        w,
        n_channels,
        z_size,
        encoder,
        decoder_linear,
        decoder,
        train_data,
        val_data,
        test_data,
        batch_size,
        dmg_size,
        encoder_hid,
    )

    vae.load("best_celebA_5_doublings")
    return vae


def conditional_samples():
    vae = load_model()
    x, _ = next(vae.val_loader)
    _, axes = plt.subplots(8, 6, figsize=(5 * 8, 5 * 6))
    ogs = axes[:, 0]
    reconstructed = axes[:, 1]

    for img, ax in zip(x, ogs):
        ax.imshow(img.permute(1, 2, 0).detach().numpy())
        ax.axis("off")

    zs, recs = vae.forward(x)
    recs = recs.mean.permute(0, 2, 3, 1)
    zs = zs.mean

    for rec, ax in zip(recs, reconstructed):
        ax.imshow(rec.detach().numpy())
        ax.axis("off")

    conv_net_positions = [
        i for i, net in enumerate(vae.decoder) if isinstance(net, nn.ConvTranspose2d)
    ]
    for i, pos in enumerate(conv_net_positions):
        axs = axes[:, 1 + i]
        damaged = vae.damage_decode(zs, pos)
        dmg_images = damaged.mean
        for ax, img in zip(axs, dmg_images):
            ax.imshow(img.permute(1, 2, 0).detach().numpy())
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        "./data/plots/damage_recovery_baseline_celebA_fully_trained.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def unconditional_samples():
    # _, axes = plt.subplots(8, 6, figsize=(5 * 8, 5 * 6))
    # reconstructed = axes[:, 0]
    vae = load_model()
    zs = vae.p_z.sample((8,))

    recs = vae.decode(zs)
    recs = recs.mean.permute(0, 2, 3, 1).detach().numpy()
    recs = [rec for rec in recs]
    all_recs = np.vstack(recs)
    # for rec, ax in zip(recs, reconstructed):
    #     ax.imshow(rec.detach().numpy())
    #     ax.axis("off")

    conv_net_positions = [
        i for i, net in enumerate(vae.decoder) if isinstance(net, nn.ConvTranspose2d)
    ]
    dmg_images_at_stages = []
    for pos in conv_net_positions:
        damaged = vae.damage_decode(zs, pos)
        dmg_images = damaged.mean.permute(0, 2, 3, 1).detach().numpy()
        dmg_images = [img for img in dmg_images]
        dmg_images_at_stages.append(np.vstack(dmg_images))

    # _, ax = plt.subplots(1, 1, figsize=(8 * 5, len(conv_net_positions) * 5))
    final_img = np.hstack([all_recs] + dmg_images_at_stages)
    # padding = np.ones((final_img.shape[0], 32, 3))
    # final_img = np.hstack((padding, final_img, padding))
    # final_img = (final_img * 255).astype(int)
    plt.imsave(
        "./data/plots/final_damage_celebA64_baseline_5_doublings.png", final_img
    )
    # ax.imshow(final_img)
    # im = Image.fromarray(final_img, "RGB")
    # im.save("./data/plots/unconditional_damage_recovery_baseline_beta_100.png")
    # ax.axis("off")
    # plt.savefig(
    #     "./data/plots/unconditioned_damage_recovery_baseline_beta_100.png",
    #     dpi=100,
    #     bbox_inches="tight",
    # )
    # # plt.show()
    # plt.close()


if __name__ == "__main__":
    unconditional_samples()
