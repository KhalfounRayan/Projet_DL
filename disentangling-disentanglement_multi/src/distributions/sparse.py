import torch
import torch.distributions as dist
from torch.distributions.utils import broadcast_all
from numbers import Number
from utils import Constants

class Sparse(dist.Distribution):
    has_rsample = False

    @property
    def mean(self):
        return (1 - self.gamma) * self.loc

    @property
    def stddev(self):
        return self.gamma * self.alpha + (1 - self.gamma) * self.scale

    def __init__(self, gamma, loc, scale, validate_args=None):
        # Broadcast the parameters to ensure compatibility
        self.loc, self.scale = broadcast_all(loc, scale)
        self.gamma = torch.tensor(gamma, device=self.loc.device) if isinstance(gamma, Number) else gamma
        self.alpha = torch.tensor(0.05).to(self.loc.device)  # Set alpha as a constant

        # Determine the batch shape
        if isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.scale.size()

        super(Sparse, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        # Generate samples based on the Sparse distribution
        shape = self._extended_shape(sample_shape)
        p = torch.bernoulli(self.gamma * torch.ones(shape).to(self.loc.device))
        res = p * self.alpha * torch.randn(shape).to(self.loc.device) + \
              (1 - p) * (self.loc + self.scale * torch.randn(shape).to(self.loc.device))
        return res

    def log_prob(self, value):
        # Compute the log probability of the value under the Sparse distribution
        res = torch.cat([
            (dist.Normal(torch.zeros_like(self.loc), self.alpha).log_prob(value) + self.gamma.log()).unsqueeze(0),
            (dist.Normal(self.loc, self.scale).log_prob(value) + (1 - self.gamma).log()).unsqueeze(0)
        ], dim=0)
        return torch.logsumexp(res, 0)
