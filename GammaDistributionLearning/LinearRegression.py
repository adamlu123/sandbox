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
    noise = np.random.normal(loc=0, scale=phi, size=[n])  # noise, with std phi

    sigma = rho * np.ones((p, p))
    np.fill_diagonal(sigma, 1)
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=n)  # X, with correlation rho

    truetheta = np.asarray([0.6, 1.2, 1.8, 2.4, 3])
    beta = np.zeros(p)
    beta[-5:] = truetheta  # beta

    Y = np.matmul(X, beta) + noise
    return Y, X, truetheta


def get_condition(Z, U, c, V, d):
    condition1 = Z>-1/c
    condition2 = torch.log(U) < 0.5 * Z**2 + d - d*V + d*torch.log(V)
    condition = condition1 * condition2
    return condition

class MarsagliaTsampler(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.log_gamma_alpha = nn.Parameter(-1.*torch.ones(size))  # TODO: see alpha init matters or not

        self.size = size
    def forward(self, batch_size):
        self.alpha = self.log_gamma_alpha.exp() + 1  # right now only for alpha > 1
        d = self.alpha - 1/3
        c = 1. / torch.sqrt(9. * d)
        Z = Normal(0, 1).sample([batch_size, self.size])
        U = Uniform(0, 1).sample([batch_size, self.size])
        V = (1+c*Z)**3

        condition = get_condition(Z, U, c, V, d).type(torch.float)
        out = condition * d*V
        processed_out = torch.stack([out[:,p][out[:,p]>0][:10] for p in range(self.size)], dim=0).t()
        # out = out[out>0]
        detached_gamma_alpha = self.alpha #.detach()
        return processed_out, detached_gamma_alpha

class SpikeAndSlabSampler(nn.Module):
    def __init__(self, p, alternative_sampler=MarsagliaTsampler):
        super(SpikeAndSlabSampler, self).__init__()
        self.p = p
        self.logalpha = nn.Parameter(torch.ones(p))
        self.alternative_sampler = alternative_sampler(size=p)

        # L0 related parameters
        self.zeta = 1.1
        self.gamma = -0.1
        self.beta = 2 / 3

    def forward(self, batch_size):
        # sample theta
        theta, detached_gamma_alpha = self.alternative_sampler(batch_size=batch_size) # shape=[batch_size, p]

        # sample z
        u = Uniform(0, 1).sample([10, self.p])  # TODO: 10 is the number of effective samples
        s = torch.sigmoid((torch.log(u / (1 - u)) + self.logalpha) / self.beta)
        z = torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)

        if not self.training:
            z = torch.clamp(torch.sigmoid(self.logalpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

        return z, theta, detached_gamma_alpha


class LinearModel(nn.Module):
    def __init__(self, p):
        super(LinearModel, self).__init__()
        self.sampler = SpikeAndSlabSampler(p=p, alternative_sampler=MarsagliaTsampler)



    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)
        self.z, self.theta, self.detached_gamma_alpha = self.sampler(batch_size=64)
        sign = 2 * Bernoulli(0.5).sample(self.theta.size()) - 1
        self.signed_theta = sign * self.theta
        self.Theta = self.z * self.signed_theta

        out = x.matmul(self.Theta.mean(dim=0))  # TODO: defer taking mean to the output
        return out

    def kl(self, phi):
        qlogp = torch.log(nlp_pdf(self.signed_theta, phi, tau=0.358)+1e-4)
        gamma = Gamma(concentration=self.detached_gamma_alpha, rate=1.)
        qlogq = np.log(0.5) + gamma.log_prob(self.theta)  # use unsigned self.theta to compute qlogq
        return (qlogq - qlogp).sum(dim=1).mean()


def loglike(y_hat, y):
    # std = y.std()
    ll = np.log(1/(np.sqrt(2*np.pi)*1)) - (y_hat-y)**2/(2*1**2)
    return ll.sum()

def train(Y, X, truetheta, phi, epoch=10000):
    Y = torch.tensor(Y, dtype=torch.float)
    linear = LinearModel(p=X.shape[0])
    optimizer = optim.SGD(linear.parameters(), lr=0.001, momentum=0.9)

    for i in range(epoch):
        optimizer.zero_grad()
        # forward pass and compute loss
        y_hat = linear(X)

        nll = -loglike(y_hat, Y)
        kl = linear.kl(phi)
        loss = nll + kl

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # print intermediet results
        if i % 500 == 0:
            sse = ((y_hat - Y) ** 2).mean().detach().numpy()
            print(y_hat[:5].tolist(), Y[:5].tolist(), Y.std())
            print('epoch {}, z min: {}, z mean: {}, non-zero: {}'.format(i, linear.z.min(), linear.z.mean(), linear.z.nonzero().shape))
            print('p={}, phi={}, loss: {}, nll:{}, kl:{}. SSE: {}'.format(X.shape[0], phi, nll, loss, kl, sse))




def main():
    n = 100
    for p in [100, 500, 1000]:
        for phi in [1, 4, 8]:
            Y, X, truetheta = generate_data(n, p, phi, rho=0, seed=1234)
            train(Y, X, truetheta, phi, epoch=10000)



if __name__=='__main__':
    main()

