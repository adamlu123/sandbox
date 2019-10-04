import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal, Uniform, Gamma
import matplotlib.pyplot as plt

def get_condition(Z, U, c, V, d):
    condition1 = Z>-1/c
    condition2 = torch.log(U) < 0.5 * Z**2 + d - d*V + d*torch.log(V)
    condition = condition1 * condition2
    # out = torch.where(condition1 and condition2, torch.ones_like(condition1), torch.zeros_like(condition1))
    return condition



class MarsagliaTsampler(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.alpha = nn.Parameter(3*torch.ones(size))

    def forward(self, batch_size):
        d = self.alpha - 1/3
        c = 1. / torch.sqrt(9. * d)
        Z = Normal(0, 1).sample([batch_size])
        U = Uniform(0, 1).sample([batch_size])
        V = (1+c*Z)**3

        condition = get_condition(Z, U, c, V, d).type(torch.float)
        out = condition * d*V
        return out

def CE_Gamma(samples, target_distribution):
    nll = -target_distribution.log_prob(samples).mean()
    # nll = -torch.log(beta**alpha/() * samples **(alpha-1.) * torch.exp(-beta * samples)).mean()
    return nll


def main():
    # TODO 1: add plot output
    epoch = 10000
    sampler = MarsagliaTsampler(size=1)
    optimizer = optim.SGD(sampler.parameters(), lr=0.001, momentum=0.9)

    for i in range(epoch):
        # compute loss
        samples = sampler(batch_size=128)
        samples = samples[samples>0]
        target_distribution = Gamma(concentration=2, rate=1)
        loss = CE_Gamma(samples, target_distribution)  # For MarsagliaTsampler, currently only supports beta=1

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print intermediet results
        if i % 1000 == 0:
            print('loss {}'.format(loss))
            path = '/extra/yadongl10/git_project/GammaLearningResult'
            plt.hist()


if __name__ == '__main__':
    main()


