import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal, Uniform, Gamma, Bernoulli
import matplotlib.pyplot as plt
import numpy as np

def nlp_pdf(x, phi, tau=0.358):
    return x**2*1/tau*(1/np.sqrt(2*np.pi*tau*phi))*torch.exp(-x**2/(2*tau*phi))

def generate_data(n, p, phi, rho, seed):
    np.random.seed(seed)
    noise = np.random.normal(loc=0, scale=np.sqrt(phi), size=[n])  # noise, with std phi

    sigma = rho * np.ones((p, p))
    np.fill_diagonal(sigma, 1)
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=n)  # X, with correlation rho

    truetheta = np.asarray([0.6, 1.2, 1.8, 2.4, 3])
    theta = np.zeros(p)
    theta[-5:] = truetheta  # beta

    Y = np.matmul(X, theta) + noise
    return Y, X, theta




class LogNormalSampler(nn.Module):
    def __init__(self, p):
        super(LogNormalSampler, self).__init__()
        self.mu = nn.Parameter(.2*torch.ones(p))
        self.logstd = nn.Parameter(.2*torch.ones(p))
        self.p = p

    def forward(self, repeat):
        epsilon = Normal(0, 1).sample([repeat, self.p])
        return (self.logstd.exp() * epsilon + self.mu).exp()


class HardConcreteSampler(nn.Module):
    """
    Sampler for Hard concrete random variable used for L0 gate
    """
    def __init__(self, p):
        super(HardConcreteSampler, self).__init__()
        self.p = p
        self.zeta, self.gamma, self.beta = 1.1, -0.1, 2/3
        self.gamma_zeta_logratio = -self.gamma / self.zeta
        self.logalpha = nn.Parameter(torch.ones(p))

    def forward(self, repeat):
        u = Uniform(0, 1).sample([repeat, self.p])
        s = torch.sigmoid((torch.log(u / (1 - u)) + self.logalpha) / self.beta)
        if self.training:
            z = torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)
        else:
            z = torch.clamp(torch.sigmoid(self.logalpha) * (self.zeta - self.gamma) + self.gamma, 0, 1).expand_as(u)
        return z


class ConcreteSampler(nn.Module):
    """
    Sampler for concrete relation of Bernoulli variable
    """
    def __init__(self, p, temperature=0.9):
        super(ConcreteSampler, self).__init__()
        self.p = p
        self.temperature = temperature
        self.logalpha = nn.Parameter(torch.ones(p))

    def forward(self, repeat):
        u = Uniform(0, 1).sample([repeat, self.p])
        s = torch.sigmoid((torch.log(u / (1 - u)) + self.logalpha) / self.temperature)
        if not self.training:
            s = self.logalpha.exp() / (1 + self.logalpha.exp())
        return s


class SpikeAndSlabSampler(nn.Module):
    """
    Add spike and slab to LogNormalSampler
    """
    def __init__(self, p, alternative_sampler=LogNormalSampler):
        super(SpikeAndSlabSampler, self).__init__()
        self.p = p
        self.alternative_sampler = alternative_sampler(p)
        self.z_sampler = HardConcreteSampler(p)
        self.mixture_weight_sampler = ConcreteSampler(p)

    def forward(self, repeat):
        theta = self.alternative_sampler(repeat)
        weight = self.mixture_weight_sampler(repeat)
        z = self.z_sampler(repeat)
        out = z * (weight * theta + (1 - weight) * theta)
        return out


class LinearModel(nn.Module):
    """
    Wrap around SpikeAndSlabSampler for a linear regression model, use Gamma distribution as variational distribution

    compare with: https://www.tandfonline.com/doi/pdf/10.1080/01621459.2015.1130634?needAccess=true
    """
    def __init__(self, p, bias=False, alternative_sampler=LogNormalSampler):
        super(LinearModel, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.ones(p[1]))

        if isinstance(p, tuple):
            p = p[0] * p[1]
        self.alternative_sampler = alternative_sampler
        self.sampler = SpikeAndSlabSampler(p=p, alternative_sampler=alternative_sampler)

    def forward(self, x):
        self.Theta = self.sampler(repeat=60)
        out = x.matmul(self.Theta.mean(dim=0).view(x.shape[1], -1)) + self.bias  # TODO: defer taking mean to the output
        return out

    def kl(self, phi):
        p = 5e-3
        qz = self.z_mean.expand_as(self.theta)
        kl_z = qz * torch.log(qz / p) + (1 - qz) * torch.log((1 - qz) / (1 - p))
        qlogp = torch.log(nlp_pdf(self.signed_theta, phi, tau=0.358)+1e-8)

        if isinstance(self.alternative_sampler, MarsagliaTsampler):
            gamma = Gamma(concentration=self.detached_gamma_alpha, rate=1.)
            qlogq = np.log(0.5) + gamma.log_prob(self.theta)  # use unsigned self.theta to compute qlogq
        elif isinstance(self.alternative_sampler, LogNormalSampler):
            qlogq = self.log_prob_alternative(self.signed_theta)

        kl_beta = qlogq - qlogp
        kl = (kl_z + qz*kl_beta).sum(dim=1).mean()
        return kl, kl_z, kl_beta

    def log_prob_alternative(self, theta):
        """
        neg entropy of the variational distribution: qlogq, 1/S*logprob(samples)
        :param theta:
        :return:
        """