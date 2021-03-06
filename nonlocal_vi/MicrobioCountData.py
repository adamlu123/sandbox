# from LinearRegression import LinearModel
from LinearRegressionFlow import LinearModel, FlowAlternative
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal, Uniform, Gamma, Bernoulli, uniform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import utils


def nlp_pdf(x, phi, tau=0.358):
    return x**2*1/tau*(1/np.sqrt(2*np.pi*tau*phi))*torch.exp(-x**2/(2*tau*phi))


def save_result(valpha, theta):
    with open('/extra/yadongl10/git_project/sandbox/nonlocal_vi/results/DDM_valpha.pkl', 'wb') as file:
        pkl.dump(valpha, file)
    with open('/extra/yadongl10/git_project/sandbox/nonlocal_vi/results/DDM_theta.pkl', 'wb') as file:
        pkl.dump(theta, file)

class LinearDiracDelta(nn.Module):
    """
    use Dirac delta mass as variational distribution
    """
    def __init__(self, p, q):
        super(LinearDiracDelta, self).__init__()
        self.p = p
        self.q = q
        init_theta = uniform.Uniform(-1, 1).sample([p, q])  #.1.*torch.ones(p, q)
        self.theta = nn.Parameter(init_theta)
        self.bias = nn.Parameter(torch.ones(q))
        self.logalpha = nn.Parameter(torch.ones(p, q))

        # L0 related parameters
        self.zeta = 1.1
        self.gamma = -0.1
        self.beta = 2 / 3
        self.gamma_zeta_logratio = -self.gamma / self.zeta

    def forward(self, x):
        # sample z
        # self.logalpha = self.logalpha0.clamp(min=-1e2, max=1e2)
        u = Uniform(0, 1).sample([10, self.p, self.q])  # TODO: 10 is the number of effective samples
        s = torch.sigmoid((torch.log(u / (1 - u)) + self.logalpha) / self.beta)
        self.z = torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)
        self.z_mean = torch.sigmoid(self.logalpha - self.beta * self.gamma_zeta_logratio)

        if not self.training:
            self.z = torch.clamp(torch.sigmoid(self.logalpha) * (self.zeta - self.gamma) + self.gamma, 0, 1).expand_as(u)

        self.Theta = self.theta * self.z.mean(dim=0)  # TODO: defer taking mean to the output
        out = x.matmul(self.Theta) + self.bias
        logalpha = self.logalpha
        return out

    def kl(self, phi):
        qz = self.z_mean.expand_as(self.theta)  # TODO: is this line necessary?
        qlogp = torch.log(nlp_pdf(self.theta, phi, tau=0.358)+1e-8)
        qlogq = 0
        kl_beta = qlogq - qlogp
        p = 1e-1
        kl_z = qz*torch.log(qz/p) + (1-qz)*torch.log((1-qz)/(1-p))
        kl = (kl_z + qz*kl_beta).sum()
        return kl, kl_z.sum(), kl_beta.sum()




def get_nll(y_pred, labels):
    delta = torch.exp(y_pred).clamp(min=1e-2, max=1e2)
    ll = torch.lgamma(labels + delta) - torch.lgamma(delta)  # shape(n,q)
    ll = ll.sum(dim=-1) + (torch.lgamma(delta.sum(dim=-1)) - torch.lgamma(
        labels.sum(dim=-1) + delta.sum(dim=-1)))
    ll = ll.mean()
    return -ll


def train(Y, X, phi, epoch=15000):
    Y = torch.tensor(Y, dtype=torch.float)
    X = torch.tensor(X, dtype=torch.float)
    # linear = LinearDiracDelta(p=X.shape[1], q=Y.shape[1])
    linear = LinearModel(p=(X.shape[1], Y.shape[1]),
                         bias=True,
                         alternative_sampler=FlowAlternative)

    optimizer = optim.SGD(linear.parameters(), lr=0.01, momentum=0.9)
    sse_list = []
    for i in range(epoch):
        linear.train()
        optimizer.zero_grad()
        # forward pass and compute loss
        y_hat = linear(X)

        nll = get_nll(y_hat, Y)
        kl, kl_z, kl_beta = linear.kl(phi)
        loss = nll + 1/10*kl

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        sse = ((y_hat - Y) ** 2).mean().detach().numpy()
        sse_list.append(sse)

        # print intermediet results
        if i % 5 == 0:
            torch.manual_seed(10)
            with torch.no_grad():
                linear.eval()
                y_hat = linear(X)
                z = linear.z.mean(dim=0).cpu().numpy()
                sse_test = ((y_hat - Y) ** 2).mean().detach().numpy()

            print('\n', y_hat[-1, :5].exp().round().tolist(), Y[-1,:5].round().tolist())
            # print('est.Thetas: {}, est z:{}'.format(linear.Theta[-5:].tolist(), z.mean(dim=0).detach().numpy().round(2)))
            print('Bacteriods: {}'.format(z.reshape(117, 30)[:, 0].max()))
            print('bias: {}'.format(linear.bias.tolist()))
            print('epoch {}, z min: {}, z mean: {}, z max: {} non-zero: {}'.format(i, z.min(), z.mean(), z.max(), None))
            print('theta min: {}, theta mean: {}, theta max: {}'.format(linear.theta.min(), linear.theta.mean(), linear.theta.max()))
            print('p={}, phi={}, loss: {}, nll:{}, kl:{}. kl_z:{}, kl_beta:{}, SSE: {}, sse_test: {}'.format(X.shape[0], phi, loss, nll, kl, kl_z, kl_beta, sse, sse_test))
            threshold = utils.search_threshold(z, 0.3)
            print('threshold', threshold)
            print('number of cov above threshold', np.sum(z > threshold))
    save_result(linear.logalpha, linear.theta)
    plt.plot(sse_list)
    plt.savefig('/extra/yadongl10/git_project/GammaLearningResult/sse.png', dpi=100)
    return linear




def main():
    real_data_dir = '/extra/yadongl10/DMVS/'
    data = {}
    data['Y'] = pd.read_csv(real_data_dir + 'adj_taxacount.txt', sep='\t').iloc[:, 1:].to_numpy()
    data['X'] = pd.read_csv(real_data_dir + 'adj_nutri.txt', sep='\t').iloc[:, 1:].to_numpy()
    print('Y shape:{}, X shape:{}'.format(data['Y'].shape, data['X'].shape))
    model = train(data['Y'], data['X'], phi=1, epoch=15000)


if __name__=='__main__':
    main()
