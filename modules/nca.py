import torch as t
import shapeguard
from einops import rearrange
from torch.nn.init import orthogonal
from torch.utils.checkpoint import checkpoint


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
        self.rnn = t.nn.GRUCell(state_dim, state_dim)

    def step(self, state: t.Tensor, rand_update_mask):
        state.sg("bc**")
        b, c, h, w = state.shape
        state = state.view_as(state)
        update = self.update_net(state)
        rnn_in = rearrange(update, "b c h w -> (b h w) c")
        rnn_hid = rearrange(state, "b c h w -> (b h w) c")
        rnn_out = self.rnn.forward(rnn_in, rnn_hid)
        new_state = rearrange(rnn_out, "(b h w) c -> b c h w", b=b, c=c, h=h, w=w)
        # new_state = (state + update * rand_update_mask)
        # new_state[:, (self.state_dim // 2):, :, :] = state[:, (self.state_dim // 2):, :, :]  # keep DNA part
        return new_state

    def forward(self, state):
        state.sg("bc22")
        states = [state]

        for j in range(self.steps_per_duplication):
            rand_update_mask = (t.rand((state.shape[0], 1, state.shape[2], state.shape[3]), device=self.device) < self.p_update).to(t.float32)
            state = checkpoint(self.step, state, rand_update_mask)
            states.append(state)

        for i in range(self.n_duplications):
            state = t.repeat_interleave(t.repeat_interleave(state, 2, dim=2), 2, dim=3)  # cell division
            states.append(state)
            for j in range(self.steps_per_duplication):
                rand_update_mask = (t.rand((state.shape[0], 1, state.shape[2], state.shape[3]), device=self.device) < self.p_update).to(t.float32)
                state = checkpoint(self.step, state, rand_update_mask)
                states.append(state)

        return states
