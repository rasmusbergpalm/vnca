import io
import math

import PIL
import numpy as np
import requests
import tqdm
from torchvision import transforms

import util
from modules.nca import MitosisNCA
import torch as t

from modules.sobel_perception import SobelPerception


class Model(t.nn.Module):
    h = w = 128
    n_duplications = 5  # 2, (4, 8, 16, 32, 64)
    steps_per_duplication = 8
    c = 16
    p = 3 * 16
    batch_size = 8
    training_iterations = 8001
    EMOJI = '🦎😀💥👁🐠🦋🐞🕸🥨🎄'

    def __init__(self):
        super().__init__()
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        update_net = t.nn.Sequential(
            t.nn.Conv2d(3 * self.c, 128, 1), t.nn.ReLU(),
            t.nn.Conv2d(128, self.c, 1)
        )
        update_net[0].bias.data.fill_(0.0)
        update_net[2].weight.data.fill_(0.0)
        update_net[2].bias.data.fill_(0.0)

        self.nca = MitosisNCA(self.h, self.w, self.c, SobelPerception(self.c), update_net, self.n_duplications, self.steps_per_duplication, 3)
        self.target = self.load_emoji('🐠', 64)
        self.optim = t.optim.Adam(self.parameters(), lr=2e-3)
        self.to(self.device)
        self.train_writer, self.test_writer = util.get_writers('hierarchical-nca')
        print(self)

    def load_image(self, url, size):
        r = requests.get(url)
        img = PIL.Image.open(io.BytesIO(r.content))

        resized = img.copy()
        resized.thumbnail((size, size), PIL.Image.ANTIALIAS)
        resized = np.float32(resized) / 255.0
        # premultiply RGB by Alpha
        resized[..., :3] *= resized[..., 3:]
        # pad to 72, 72
        resized = t.tensor(resized, device=self.device).permute(2, 0, 1)
        pad = (self.h - size) // 2
        resized = t.nn.functional.pad(resized, [pad, pad, pad, pad], mode="constant", value=0)

        return resized

    def load_emoji(self, emoji, size):
        code = hex(ord(emoji))[2:].lower()
        url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true' % code
        return self.load_image(url, size)

    def train_batch(self, state):
        # (b, c, h, w)
        self.optim.zero_grad()
        states = self.nca(state)
        loss = self.loss(states[-1], self.target).mean()
        loss.backward()
        for p in self.parameters():  # grad norm
            p.grad /= (t.norm(p.grad) + 1e-8)
        self.optim.step()
        return [state.detach() for state in states], loss.item()

    def _to_rgb(self, x):
        # assume rgb premultiplied by alpha
        rgb = x[:, :3, :, :]  # 0,0,0
        a = t.clamp(x[:, 3:4, :, :], 0.0, 1.0)  # 1.0
        im = 1.0 - a + rgb  # (1-1+0) = 0, (1-0+0) = 1
        im = t.clamp(im, 0, 1)
        return im

    def non_pool_train(self):
        seed = t.zeros(self.c, self.h, self.w, device=self.device)
        seed[3:, self.h // 2 - 1:self.h // 2 + 1, self.w // 2 - 1:self.w // 2 + 1] = 1.0  # the middle 2x2
        for i in tqdm.tqdm(range(self.training_iterations)):
            batch = seed.unsqueeze(0)  # just a single batch for now
            outputs, loss = self.train_batch(batch)  # (steps, 1, C, H, W)
            outputs = t.cat(outputs, dim=0)  # (steps, C, H, W)
            zoomed = t.zeros_like(outputs)

            for j in range(self.n_duplications + 1):
                # 0-8, 9-17, 18-26, ...
                level = outputs[j * 9:(j + 1) * 9, :, :, :]
                center_crop_zoom = transforms.Compose([
                    transforms.CenterCrop(2 ** (j + 2)),  # 4, 8, 16, ..., 128
                    transforms.Resize(128, interpolation=transforms.InterpolationMode.NEAREST),
                ])
                zoomed[j * 9:(j + 1) * 9, :, :, :] = center_crop_zoom(level)

            self.train_writer.add_scalar('log10(loss)', math.log10(loss), i)
            self.train_writer.add_images("outputs", self._to_rgb(outputs), i, dataformats='NCHW')
            self.train_writer.add_images("zoomed", self._to_rgb(zoomed), i, dataformats='NCHW')

    def loss(self, state, target):
        return ((state[:, :4] - target) ** 2).mean(dim=(1, 2, 3))


if __name__ == '__main__':
    model = Model()
    model.non_pool_train()
