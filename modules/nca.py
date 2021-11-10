import random

import torch as t
from torch.utils.checkpoint import checkpoint


class NCA(t.nn.Module):

    def __init__(self, update_net: t.nn.Module, min_steps, max_steps, p_update=0.5):
        super().__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.update_net = update_net
        self.p_update = p_update

    def step(self, state, rand_update_mask):
        # pre_alive_mask = self.alive_mask(state)

        update = self.update_net(state)
        state = (state + update * rand_update_mask)

        # post_alive_mask = self.alive_mask(state)
        state = state  # * (post_alive_mask * pre_alive_mask)

        return state

    """
    def alive_mask(self, state):
        x = max_pool2d(state[:, 0:1], kernel_size=(3, 3), stride=1, padding=1)
        hard = (t.sigmoid(x - 6.0) > 0.1).to(t.float32)
        soft = x  # t.sigmoid(x - 6.0 - t.logit(t.tensor(0.1)))
        out = hard + soft - soft.detach()

        return out
    """

    def forward(self, state):
        states = [state]

        for j in range(random.randint(self.min_steps, self.max_steps)):
            rand_update_mask = (t.rand((state.shape[0], 1, state.shape[2], state.shape[3]), device=self.device) < self.p_update).to(t.float32)
            state = checkpoint(self.step, state, rand_update_mask)
            states.append(state)

        return states
