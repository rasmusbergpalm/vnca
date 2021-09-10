import torch as t
from torch.distributions import Distribution


def logistic_cdf(x: t.Tensor, loc: t.Tensor, logscale: t.Tensor) -> t.Tensor:
    a = (x - loc) / logscale.exp()
    return t.sigmoid(a)


def discrete_logistic_pmf(x: t.Tensor, loc: t.Tensor, logscale: t.Tensor, min, max, step):
    """
    Computes the PMF of the discrete logistic distribution defined over T.arange(min, max+step, step).
    """
    assert min <= x.min() and x.max() <= max
    interval = t.scalar_tensor(step / 2, device=x.device)
    cdf_inf_to_min = logistic_cdf(min, loc, logscale)
    cdf_max_to_inf = 1 - logistic_cdf(max, loc, logscale)
    cdf_plus = logistic_cdf(x + interval, loc, logscale)
    cdf_minus = logistic_cdf(x - interval, loc, logscale)
    cdf_x_plus_minus_interval = cdf_plus - cdf_minus

    return t.where(
        x == min,
        cdf_inf_to_min,
        t.where(
            x == max,
            cdf_max_to_inf,
            cdf_x_plus_minus_interval
        )
    )


def log_discrete_logistic_pmf(x: t.Tensor, loc: t.Tensor, logscale: t.Tensor, min, max, step):
    """
    Computes the log PMF of the discrete logistic distribution defined over T.arange(min, max+step, step).
    """
    p = discrete_logistic_pmf(x, loc, logscale, min, max, step)
    # p underflows for very small values. In those cases use an approximation of the log PMF instead: log(PDF of bin center * bin size)
    a = -(x - loc) / logscale.exp()
    logpmf_square_approx = a - logscale - 2 * t.nn.functional.softplus(a) + t.log(t.scalar_tensor(step, device=x.device))
    return t.where(p > 1e-5, t.log(t.clamp_min(p, 1e-12)), logpmf_square_approx)


class DiscreteLogistic(Distribution):

    def __init__(self, loc: t.Tensor, logscale: t.Tensor, min, max, step):
        assert loc.shape == logscale.shape
        super().__init__(batch_shape=loc.shape, validate_args=False)
        self.loc = loc
        self.logscale = logscale
        self.min = min
        self.max = max
        self.step = step

    @property
    def mean(self):
        return self.loc

    def log_prob(self, x):
        return log_discrete_logistic_pmf(x, self.loc, self.logscale, self.min, self.max, self.step)


if __name__ == '__main__':
    dl = DiscreteLogistic(t.tensor(0.5), t.tensor(-3), 0, 1, 1 / 256)
    x = t.tensor([8/256, 128/256])
    print(dl.log_prob(x))
    print(dl.mean)
    print(dl.batch_shape)
