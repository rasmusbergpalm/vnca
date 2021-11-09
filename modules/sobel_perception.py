import torch as t


class SobelConv(t.nn.Module):

    def __init__(self, channels_in, device):
        super().__init__()
        self.channels_in = channels_in
        t.nn.Parameter()
        self.sobel_x = t.tensor([[-1.0, 0.0, +1.0],
                                 [-2.0, 0.0, +2.0],
                                 [-1.0, 0.0, +1.0]], device=device).unsqueeze(0).unsqueeze(0).expand((channels_in, 1, 3, 3)) / 8.0  # (out, in, h, w)
        self.sobel_y = self.sobel_x.permute(0, 1, 3, 2)

    def forward(self, state):
        # Convolve sobel filters with states
        # in x, y and channel dimension.
        grad_x = t.conv2d(state, self.sobel_x, groups=self.channels_in, padding=1)
        grad_y = t.conv2d(state, self.sobel_y, groups=self.channels_in, padding=1)
        # Concatenate the cell’s state channels,
        # the gradients of channels in x and
        # the gradient of channels in y.
        return t.cat([state, grad_x, grad_y], dim=1)
