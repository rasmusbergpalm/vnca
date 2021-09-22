import torch as t


class Residual(t.nn.Module):
    def __init__(self, *args: t.nn.Module):
        super().__init__()
        self.delegate = t.nn.Sequential(*args)

    def forward(self, inputs):
        return self.delegate(inputs) + inputs
