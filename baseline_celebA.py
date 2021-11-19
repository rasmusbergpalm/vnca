import os

from torch import nn
from torchvision import transforms, datasets

from modules.vae import VAE
from train import train


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
h = w = 32
n_channels = 3


encoder = nn.Sequential(
    nn.Conv2d(n_channels, encoder_hid * 2 ** 0, filter_size, padding=pad),
    nn.ELU(),  # (bs, 32, h, w)
    nn.Conv2d(
        encoder_hid * 2 ** 0, encoder_hid * 2 ** 1, filter_size, padding=pad, stride=2
    ),
    nn.ELU(),  # (bs, 64, h//2, w//2)
    nn.Conv2d(
        encoder_hid * 2 ** 1, encoder_hid * 2 ** 2, filter_size, padding=pad, stride=2
    ),
    nn.ELU(),  # (bs, 128, h//4, w//4)
    nn.Conv2d(
        encoder_hid * 2 ** 2, encoder_hid * 2 ** 3, filter_size, padding=pad, stride=2
    ),
    nn.ELU(),  # (bs, 256, h//8, w//8)
    nn.Conv2d(
        encoder_hid * 2 ** 3, encoder_hid * 2 ** 4, filter_size, padding=pad, stride=2
    ),
    nn.ELU(),  # (bs, 512, h//16, w//16),
    nn.Flatten(),  # (bs, 512*h//16*w//16)
    nn.Linear(encoder_hid * (2 ** 4) * h // 16 * w // 16, 2 * z_size),
)

decoder_linear = nn.Sequential(
    nn.Linear(z_size, encoder_hid * (2 ** 4) * h // 16 * w // 16)
)

decoder = nn.Sequential(
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
        encoder_hid * 2 ** 0,
        n_mixtures * 10,
        filter_size,
        padding=pad,
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
vae.eval_batch()
train(vae, n_updates=100_000, eval_interval=100)
vae.test(128)
