import torch as t
import shapeguard


class MitosisNCA(t.nn.Module):

    def __init__(self, h, w, state_dim, update_net: t.nn.Module, n_duplications, steps_per_duplication, alive_channel):
        super().__init__()
        self.h = h
        self.w = w
        self.n_duplications = n_duplications
        self.steps_per_duplication = steps_per_duplication
        self.state_dim = state_dim
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.update_net = update_net
        self.alive_channel = alive_channel

    def alive_mask(self, state):
        state.sg("bchw")
        alive = (t.max_pool2d(state[:, self.alive_channel:self.alive_channel + 1, :, :], 3, stride=1, padding=1) > 0.1).to(t.float32)
        return alive

    def step(self, state):
        state.sg("bchw")
        pre_alive = self.alive_mask(state)

        update = self.update_net(state)
        rand_mask = t.randint(0, 2, (state.shape[0], 1, self.h, self.w), dtype=t.float32, device=self.device)
        state = state + rand_mask * update

        post_alive = self.alive_mask(state)
        alive_mask = pre_alive * post_alive
        return state * alive_mask

    def forward(self, state):
        state.sg("bchw")
        states = [state]

        for j in range(self.steps_per_duplication):
            state = self.step(state)
            states.append(state)

        for i in range(self.n_duplications):
            state = t.repeat_interleave(t.repeat_interleave(state, 2, dim=2), 2, dim=3)  # cell division
            state = state[:, :, self.h // 2: self.h // 2 + self.h, self.w // 2: self.w // 2 + self.w]  # cut out middle (h, w)
            states.append(state)
            for j in range(self.steps_per_duplication):
                state = self.step(state)
                states.append(state)

        return states
