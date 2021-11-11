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

    def step(self, state):
        rand_update_mask = (t.rand((state.shape[0], 1, state.shape[2], state.shape[3]), device=self.device) < self.p_update).to(t.float32)

        update = self.update_net(state)
        state = (state + update * rand_update_mask)

        return state

    def _multi_step(self, state, n_steps):
        states = [state]
        for _ in range(n_steps):
            states.append(self.step(states[-1]))

        return states[1:]

    def forward(self, state):
        n_steps = random.randint(self.min_steps, self.max_steps)
        multi_step_size = 8
        n_multi_steps, remainder_steps = n_steps // multi_step_size, n_steps % multi_step_size

        states = [state]
        for j in range(n_multi_steps):
            states += checkpoint(self._multi_step, states[-1], multi_step_size)

        if remainder_steps > 0:
            states += checkpoint(self._multi_step, states[-1], remainder_steps)

        return states
