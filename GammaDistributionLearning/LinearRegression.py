import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal, Uniform, Gamma
import matplotlib.pyplot as plt
import numpy as np
import MarsagliaTsampler


def generate_data(n, p, phi, rho, seed):
    np.random.seed(seed)
    noise = np.random.normal(loc=0, scale=phi, size=[n, p])  # noise, with std phi

    sigma = rho * np.ones((p, p))
    np.fill_diagonal(sigma, 1)
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=2)  # X, with correlation rho

    truetheta = np.asarray([0.6, 1.2, 1.8, 2.4, 3])
    beta = np.zeros(p)
    beta[-5:] = truetheta  # beta

    Y = X * beta + noise
    return Y, X, truetheta


class SpikeAndSlabSampler(nn.module):
    def __init__(self, p, alternative_sampler):
        self.p = p
        self.logalpha = nn.Parameter(torch.ones(p))
        self.alternative_sampler = alternative_sampler(size=p)

        # L0 related parameters
        self.zeta = 1.1
        self.gamma = -0.1
        self.beta = 2 / 3

    def forward(self, batch_size):
        # sample theta
        theta = self.alternative_sampler(batch_size=batch_size) # shape=[batch_size, p]

        # sample z
        u = Uniform(0, 1).sample([batch_size,self.p])
        s = torch.sigmoid((torch.log(u / (1 - u)) + self.logalpha) / self.beta)
        z = torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)

        if not self.training:
            z = torch.clamp(torch.sigmoid(self.logalpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

        return z * theta


class LinearModel(nn.module):
    def __init__(self, p):
        self.sampler = SpikeAndSlabSampler(size=p, alternative_sampler=MarsagliaTsampler)

    def forward(self, x):
        theta = self.sampler(batch_size=32).mean(axis=0)  # TODO: defer taking mean to the output
        out = x * theta
        return out

    def loglike(self, y_hat, y):
        std = y.std()
        return torch.log(1/(torch.sqrt(2*np.pi)*std)) -(y_hat-y)**2/(2*std**2)

    def kl(self, ):


    def elbo(self, y_hat, y):
        return loglike(y_hat, y) - kl


# def elbo(y_hat, y):
#     std = y.std()
#     loglike = torch.log(1/(torch.sqrt(2*np.pi)*std)) -(y_hat-y)**2/(2*std**2)
#     kl =
#     return likelihood + kl

def train(Y, X, truetheta, epoch=10000):
    linear = LinearModel(p=X.shape[0])


    optimizer = optim.SGD(linear.parameters(), lr=0.001, momentum=0.9)

    for i in range(epoch):
        y_hat = linear(X)
        loss = -linear.elbo(y_hat, Y)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print intermediet results
        if i % 500 == 0:
            sse = ((y_hat - Y) ** 2).mean()
            print('p={}, loss: {}, SSE: {}'.format(X.shape[0], loss, sse))




def main():
    n = 100
    for p in [100, 500, 1000]:
        for phi in [1, 4, 8]:
            Y, X, truetheta = generate_data(n, p, phi, rho=0, seed=1234)
            train(Y, X, truetheta, epoch=10000)


