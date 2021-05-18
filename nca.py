import io
import math
import random

import PIL.Image
import numpy as np
import requests
import torch as t
import tqdm

import shapeguard
import util


class NCA(t.nn.Module):
    h = w = 72
    c = 16
    p = 3 * 16
    grow_range = (64, 96)
    batch_size = 8
    training_iterations = 8001
    EMOJI = '🦎😀💥👁🐠🦋🐞🕸🥨🎄'

    def __init__(self):
        super().__init__()
        self.device = "cuda" if t.cuda.is_available() else "cpu"

        self.update_net = t.nn.Sequential(
            t.nn.Conv2d(self.p, 128, 1), t.nn.ReLU(),
            t.nn.Conv2d(128, self.c, 1)
        )
        self.update_net[0].bias.data.fill_(0.0)
        self.update_net[2].weight.data.fill_(0.0)
        self.update_net[2].bias.data.fill_(0.0)
        self.sobel_x = t.tensor([[-1.0, 0.0, +1.0],
                                 [-2.0, 0.0, +2.0],
                                 [-1.0, 0.0, +1.0]], device=self.device).unsqueeze(0).unsqueeze(0).expand((self.c, 1, 3, 3)) / 8.0  # (out, in, h, w)
        self.sobel_y = self.sobel_x.permute(0, 1, 3, 2)
        self.target = self.load_emoji('🦎').to(self.device)
        self.optim = t.optim.Adam(self.parameters(), lr=2e-3)
        self.to(self.device)
        self.train_writer, self.test_writer = util.get_writers('hierarchical-nca')
        print(self)

    def load_image(self, url):
        r = requests.get(url)
        img = PIL.Image.open(io.BytesIO(r.content))
        img.thumbnail((40, 40), PIL.Image.ANTIALIAS)
        img = np.float32(img) / 255.0
        # premultiply RGB by Alpha
        img[..., :3] *= img[..., 3:]
        # pad to 72, 72
        img = t.tensor(img).permute(2, 0, 1).sg((4, 40, 40))
        img = t.nn.functional.pad(img, [16, 16, 16, 16], mode="constant", value=0)

        return img

    def load_emoji(self, emoji):
        code = hex(ord(emoji))[2:].lower()
        url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true' % code
        return self.load_image(url)

    def perceive(self, state_grid):
        state_grid.sg(("b", self.c, self.h, self.w))
        # Convolve sobel filters with states
        # in x, y and channel dimension.
        grad_x = t.conv2d(state_grid, self.sobel_x, groups=16, padding=1)
        grad_y = t.conv2d(state_grid, self.sobel_y, groups=16, padding=1)
        # Concatenate the cell’s state channels,
        # the gradients of channels in x and
        # the gradient of channels in y.
        return t.cat([state_grid, grad_x, grad_y], dim=1).sg(("b", self.p, self.h, self.w))

    def update(self, state_grid):
        perception_grid = self.perceive(state_grid)
        return self.update_net(perception_grid).sg(("b", self.c, self.h, self.w))

    def alive_mask(self, state_grid):
        state_grid.sg(("b", self.c, self.h, self.w))
        # Take the alpha channel as the measure of “life”.
        alive = (t.max_pool2d(state_grid[:, 3:4, :, :], 3, stride=1, padding=1) > 0.1).to(t.float32)
        return alive

    def step(self, state_grid):
        state_grid.sg(("b", self.c, self.h, self.w))
        pre_alive = self.alive_mask(state_grid)

        update_grid = self.update(state_grid)
        rand_mask = t.randint(0, 2, (state_grid.shape[0], 1, self.h, self.w), dtype=t.float32, device=self.device)
        state_grid = state_grid + rand_mask * update_grid

        post_alive = self.alive_mask(state_grid)
        alive_mask = pre_alive * post_alive
        return state_grid * alive_mask

    def loss(self, state):
        return ((state[:, :4] - self.target) ** 2).mean(dim=(1, 2, 3))

    def train_batch(self, state):
        self.optim.zero_grad()
        for _ in range(random.randint(*self.grow_range)):
            state = self.step(state)
        loss = self.loss(state).mean()
        loss.backward()
        for p in self.parameters():  # grad norm
            p.grad /= (t.norm(p.grad) + 1e-8)
        self.optim.step()
        return state.detach(), loss.item()

    def _to_rgb(self, x):
        # assume rgb premultiplied by alpha
        rgb = x[:, :3, :, :]  # 0,0,0
        a = t.clamp(x[:, 3:4, :, :], 0.0, 1.0)  # 1.0
        im = 1.0 - a + rgb  # (1-1+0) = 0, (1-0+0) = 1
        im = t.clamp(im, 0, 1)
        return im

    def pool_training(self):
        # Set alpha and hidden channels to (1.0).
        pool_size = 1024
        seed = t.zeros(self.c, self.h, self.w, device=self.device)
        seed[3:, self.h // 2, self.w // 2] = 1.0  # rgb=0, alpha=1 = black
        pool = t.stack([seed] * pool_size)
        for i in tqdm.tqdm(range(self.training_iterations)):
            idxs = random.sample(range(len(pool)), self.batch_size)
            batch = pool[idxs]
            # Sort by loss, descending.

            sort_idx = t.argsort(self.loss(batch), descending=True)
            batch = batch[sort_idx]
            # Replace the highest-loss sample with the seed.
            batch[0] = seed
            # Perform training.
            outputs, loss = self.train_batch(batch)
            # Place outputs back in the pool.
            pool[idxs] = outputs

            self.train_writer.add_scalar('log10(loss)', math.log10(loss), i)
            self.train_writer.add_images("batch", self._to_rgb(batch), i, dataformats='NCHW')
            self.train_writer.add_images("outputs", self._to_rgb(outputs), i, dataformats='NCHW')


if __name__ == '__main__':
    nca = NCA()
    nca.pool_training()
