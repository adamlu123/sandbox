import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal, Uniform, Gamma, Bernoulli
import matplotlib.pyplot as plt
import numpy as np


def nlp_pdf(x, phi, tau=0.358):
    return x**2*1/tau*(1/np.sqrt(2*np.pi*tau*phi))*torch.exp(-x**2/(2*tau*phi))


def nlp_log_pdf(x, phi, tau=0.358):
    return (x**2).clamp(min=1e-2).log() - np.log(tau) - np.log(np.sqrt(2*np.pi*tau*phi)) - x**2/(2*tau*phi)


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
        qz = torch.sigmoid(self.logalpha - self.beta * self.gamma_zeta_logratio)
        u = Uniform(0, 1).sample([repeat, self.p])
        s = torch.sigmoid((torch.log(u / (1 - u)) + self.logalpha) / self.beta)
        if self.training:
            z = torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)
        else:
            z = torch.clamp(torch.sigmoid(self.logalpha) * (self.zeta - self.gamma) + self.gamma, 0, 1).expand_as(u)
        return z, qz


class FlowAlternative(nn.Module):
    """
    Independent component-wise flow
    """
    def __init__(self, p):
        super(FlowAlternative, self).__init__()
        self.p = p

    def forward(self, repeat):
        epsilon = Normal(0, 1).sample([repeat, self.p])
        return epsilon

class SpikeAndSlabSampler(nn.Module):
    """
    Add spike and slab to LogNormalSampler
    """
    def __init__(self, p, alternative_sampler=FlowAlternative):
        super(SpikeAndSlabSampler, self).__init__()
        self.p = p
        self.alternative_sampler = alternative_sampler(p)
        self.z_sampler = HardConcreteSampler(p)

    def forward(self, repeat):
        theta, epsilon, mu, logstd = self.alternative_sampler(repeat)
        z, qz = self.z_sampler(repeat)
        out = z * theta
        return out, epsilon, mu, logstd, z, qz


class LinearModel(nn.Module):
    """
    Wrap around SpikeAndSlabSampler for a linear regression model, use Gamma distribution as variational distribution
    compare with: https://www.tandfonline.com/doi/pdf/10.1080/01621459.2015.1130634?needAccess=true
    """
    def __init__(self, p, bias=False, alternative_sampler=FlowAlternative):
        super(LinearModel, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.ones(p[1]))

        if isinstance(p, tuple):
            p = p[0] * p[1]
        self.alternative_sampler = alternative_sampler
        self.sampler = SpikeAndSlabSampler(p=p, alternative_sampler=alternative_sampler)

    def forward(self, x):
        self.Theta, self.signed_theta, self.epsilon, self.mu, self.logstd, self.weight, self.z, self.qz = self.sampler(repeat=100)
        out = x.matmul(self.Theta.mean(dim=0).view(x.shape[1], -1))  # TODO: defer taking mean to the output
        return out.squeeze()

    def kl(self, phi):
        p = 1e-3
        qz = self.qz.expand_as(self.Theta)
        kl_z = qz * torch.log(qz / p) + (1 - qz) * torch.log((1 - qz) / (1 - p))
        # qlogp = torch.log(nlp_pdf(self.signed_theta, phi, tau=0.358)+1e-5)   # shape=(60, p)
        qlogp = nlp_log_pdf(self.signed_theta, phi, tau=0.358).clamp(min=np.log(1e-10))
        qlogq = get_qlogq_robust(self.signed_theta, self.epsilon, self.mu, self.logstd, self.weight)

        kl_beta = qlogq - qlogp
        kl = (kl_z + qz*kl_beta).sum(dim=1).mean()
        return kl, qlogp.mean(), qlogq.mean()

