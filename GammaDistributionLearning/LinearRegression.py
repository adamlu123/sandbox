import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal, Uniform, Gamma
import matplotlib.pyplot as plt
import numpy as np
import MarsagliaTsampler


def generate_data(n, p, phi, seed):
    np.random.seed(seed)


    return Y, X, truebeta, selected_id


class LinearModel(nn.module):
    def __init__(self, p):
        self.sampler = MarsagliaTsampler(size=p)
    def forward(self, x):
        theta = self.sampler(batch_size=32).mean(axis=0)
        out = x * theta
        return out


def elbo(y_hat, Y):

    return likelihood + kl

def train(Y, X, truebeta, selected_id, epoch=10000):
    linear = LinearModel(p=X.shape[0])


    optimizer = optim.SGD(linear.parameters(), lr=0.001, momentum=0.9)

    for i in range(epoch):
        y_hat = linear(X)
        loss = -elbo(y_hat, Y)

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
            Y, X, truebeta, selected_id = generate_data(n, p, phi, seed=1234)
            train(Y, X, truebeta, selected_id, epoch=10000)
