import torch as t
import shapeguard


class MitosisNCA(t.nn.Module):

    def __init__(self, h, w, state_dim, update_net: t.nn.Module, n_duplications, steps_per_duplication, p_update=0.5):
        super().__init__()
        self.h = h
        self.w = w
        self.n_duplications = n_duplications
        self.steps_per_duplication = steps_per_duplication
        self.state_dim = state_dim
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.update_net = update_net
        self.p_update = p_update

    def step(self, state):
        state.sg("bc**")
        update = self.update_net(state)
        new_state = (state + update)
        new_state[:, (self.state_dim // 2):, :, :] = state[:, (self.state_dim // 2):, :, :]  # keep DNA part
        return new_state

    def forward(self, state):
        state.sg("bc22")
        states = [state]

        for j in range(self.steps_per_duplication):
            state = t.utils.checkpoint.checkpoint(self.step, state)
            states.append(state)

        for i in range(self.n_duplications):
            state = t.repeat_interleave(t.repeat_interleave(state, 2, dim=2), 2, dim=3)  # cell division
            states.append(state)
            for j in range(self.steps_per_duplication):
                state = t.utils.checkpoint.checkpoint(self.step, state)
                states.append(state)

        return states
