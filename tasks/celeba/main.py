import os

from torch import nn
from torch.nn import DataParallel
from torchvision import transforms, datasets

from modules.dml import DiscretizedMixtureLogitsDistribution
from modules.vnca import VNCA
from train import train

if __name__ == "__main__":
    z_size = 128
    nca_hid = 256
    n_mixtures = 1
    batch_size = 128
    dmg_size = 16

    filter_size = 5
    pad = filter_size // 2
    encoder_hid = 32
    h = w = 32
    n_channels = 3


    def state_to_dist(state):
        return DiscretizedMixtureLogitsDistribution(n_mixtures, state[:, :n_mixtures * 10, :, :])


    encoder = nn.Sequential(
        nn.Conv2d(n_channels, encoder_hid * 2 ** 0, filter_size, padding=pad), nn.ELU(),  # (bs, 32, h, w)
        nn.Conv2d(encoder_hid * 2 ** 0, encoder_hid * 2 ** 1, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 64, h//2, w//2)
        nn.Conv2d(encoder_hid * 2 ** 1, encoder_hid * 2 ** 2, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 128, h//4, w//4)
        nn.Conv2d(encoder_hid * 2 ** 2, encoder_hid * 2 ** 3, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 256, h//8, w//8)
        nn.Conv2d(encoder_hid * 2 ** 3, encoder_hid * 2 ** 4, filter_size, padding=pad, stride=2), nn.ELU(),  # (bs, 512, h//16, w//16),
        nn.Flatten(),  # (bs, 512*h//16*w//16)
        nn.Linear(encoder_hid * (2 ** 4) * h // 16 * w // 16, 2 * z_size),
    )

    update_net = nn.Sequential(
        nn.Conv2d(z_size, nca_hid, 3, padding=1),
        nn.ELU(),
        nn.Conv2d(nca_hid, nca_hid, 1),
        nn.ELU(),
        nn.Conv2d(nca_hid, nca_hid, 1),
        nn.ELU(),
        nn.Conv2d(nca_hid, z_size, 1, bias=False)
    )
    update_net[-1].weight.data.fill_(0.0)

    # encoder = DataParallel(encoder)
    # update_net = DataParallel(update_net)

    data_dir = os.environ.get('DATA_DIR') or "data"
    tp = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])
    train_data, val_data, test_data = [datasets.CelebA(data_dir, split=split, download=True, transform=tp) for split in ["train", "valid", "test"]]

    vnca = VNCA(h, w, n_channels, z_size, encoder, update_net, train_data, val_data, test_data, state_to_dist, batch_size, dmg_size)
    vnca.eval_batch()
    train(vnca, n_updates=100_000, eval_interval=100)
    vnca.test(128)
