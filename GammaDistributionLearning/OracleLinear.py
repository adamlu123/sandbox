import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal, Uniform, Gamma, Bernoulli
import numpy as np


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


class OracleModel(nn.Module):
    def __init__(self):
        super(OracleModel, self).__init__()
        self.theta = nn.Parameter(torch.zeros(5))

    def forward(self, x):
        return x[:, -5:].mm(self.theta.view(-1, 1)).squeeze()  # self.theta.expand_as(x[:, -5:])


def loglike(y_hat, y):
    ll = - (y_hat - y) ** 2 / (2 * 1 ** 2)  # + np.log(1/(np.sqrt(2*np.pi)*1))
    return ll.sum()


def train(Y, X, truetheta, phi, epoch=10000):
    Y = torch.tensor(Y, dtype=torch.float)
    X = torch.tensor(X, dtype=torch.float)
    x = X[:, -5:]
    mle = (x.t().mm(x)).inverse().mm(x.t()).mm(Y.view(-1, 1))
    print('MLE is:', mle)

    linear = OracleModel()
    optimizer = optim.SGD(linear.parameters(), lr=0.001, momentum=0.9)

    for i in range(epoch):
        optimizer.zero_grad()
        # forward pass and compute loss
        y_hat = linear(X)

        nll = -loglike(y_hat, Y)
        loss = nll

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # print intermediet results
        if i % 500 == 0:
            sse = ((y_hat - Y) ** 2).mean().detach().numpy()
            print(linear.theta.detach().numpy())
            print('p={}, phi={}, loss: {}, nll:{}, MSE: {}'.format(X.shape[0], phi, nll, loss, sse))


def main():
    n = 100
    for p in [100, 500, 1000]:
        for phi in [1, 4, 8]:
            Y, X, truetheta = generate_data(n, p, phi, rho=0, seed=1234)
            train(Y, X, truetheta, phi, epoch=10000)


if __name__ == '__main__':
    main()
