import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Function
import pickle as pkl
from torch.distributions import Gamma, Beta
from torch.distributions.kl import kl_divergence
import numpy as np

hp = {
    'data_folder': '/extra/yadongl10/data/MNIST/',
    'device': 'cuda',
    'epochs': 100000
}


def prepare_data(subset, hp):
    if subset == 'train':
        data = np.loadtxt(hp['data_folder'] + 'binarized_mnist_train_small.txt')
        # data = data[:5000, :]
        print(data.shape)
    elif subset == 'test':
        data = np.loadtxt(hp['data_folder'] + 'binarized_mnist_test.amat')
        data = data[:1000, :]
        print(data.shape)

    data = torch.tensor(data, dtype=torch.float32).to(hp['device'])
    data = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(data, batch_size=5000, shuffle=True)
    return data_loader


def update_samples(logit_z, log_w):
    logit_z = logit_z + logit_z.grad * 1e-4
    log_w = log_w + log_w.grad * 1e-4
    return logit_z, log_w


def log_prob_beta(theta, eta, value):
    return (theta - 1) * torch.log(value) + (eta - 1) * torch.log(1 - value) + \
           torch.lgamma(theta + eta) - torch.lgamma(theta) - torch.lgamma(eta)


def log_prob_gamma(alpha, beta, value):
    return alpha * torch.log(beta) + (alpha - 1) * torch.log(value) - beta * value - torch.lgamma(alpha)


class BetaGammaSampler(nn.Module):
    def __init__(self):
        super(BetaGammaSampler, self).__init__()
        self.logalpha = nn.Parameter(np.log(1)*torch.ones(100, 784))
        self.logbeta = nn.Parameter(np.log(15)*torch.ones(100, 784))
        self.logtheta = nn.Parameter(2*torch.ones(5000, 100))
        self.logeta = nn.Parameter(2*torch.ones(5000, 100))

    def sample(self):
        z = Beta(concentration0=self.logtheta.exp().detach(), concentration1=self.logeta.exp().detach()).sample()
        logit_z = torch.log(z / (1 - z))
        w = Gamma(concentration=self.logalpha.exp().detach(), rate=self.logbeta.exp().detach()).sample()
        log_w = w.log()
        return logit_z, log_w

    def get_kl(self):
        gamma_q = Gamma(concentration=self.logalpha.exp(), rate=self.logbeta.exp())
        gamma_p = Gamma(0.1 * torch.ones_like(self.logalpha), 0.3 * torch.ones_like(self.logalpha))
        beta_q = Beta(self.logtheta.exp(), self.logeta.exp())
        beta_p = Beta(torch.ones_like(self.logtheta), torch.ones_like(self.logtheta))
        kl = kl_divergence(beta_q, beta_p).sum() + kl_divergence(gamma_q, gamma_p).sum()
        return kl

    def forward(self, logit_z_updated, log_w_updated):
        z_updated = torch.sigmoid(logit_z_updated)
        ll_gamma = log_prob_gamma(self.logalpha.exp(), self.logbeta.exp(), log_w_updated.exp())
        ll_beta = log_prob_beta(self.logtheta.exp(), self.logeta.exp(), z_updated)
        kl = self.get_kl()
        return ll_gamma.sum() + ll_beta.sum() - kl, kl.detach()


class BetaGammaMF(nn.Module):
    def __init__(self):
        super(BetaGammaMF, self).__init__()

    def forward(self, logit_z, log_w):
        w = log_w.exp()
        out = torch.sigmoid(logit_z.matmul(w))
        return out

def get_ll(out, data):
    return data * (out + 1e-10).log() + (1 - data) * (1 - out + 1e-10).log()


def train(model, sampler, optimizer, train_loader):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data[0]
        optimizer.zero_grad()

        logit_z, log_w = sampler.sample()
        out = model(logit_z.requires_grad_(), log_w.requires_grad_())
        ll = get_ll(out, data).sum()

        ll.backward()
        logit_z_updated, log_w_updated = update_samples(logit_z, log_w)
        logit_z.grad.zero_()
        log_w.grad.zero_()
        ll = ll.detach()

        elbo_biased, kl = sampler(logit_z_updated, log_w_updated)  # used for compute gradients

        loss = - elbo_biased
        loss.backward()
        optimizer.step()

        elbo = ll - kl  # true elbo
        return elbo, kl, ll


def main(hp):
    # Define model
    print('start define model')
    model = BetaGammaMF().to(hp['device'])
    sampler = BetaGammaSampler().to(hp['device'])

    # prepare data
    print('start loading data')
    train_loader = prepare_data('train', hp)
    test_loader = prepare_data('test', hp)

    # define opt
    # optimizer_model = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_sampler = torch.optim.Adam(sampler.parameters(), lr=0.001)

    # training
    print('start training')
    elbo_list = []
    kl_list = []
    ll_list = []
    for epoch in range(hp['epochs']):
        elbo, kl, ll = train(model, sampler, optimizer_sampler, train_loader)
        elbo_list.append(elbo)
        kl_list.append(kl)
        ll_list.append(ll)
        if epoch % 50 == 0:
            print('epoch: {}, elbo: {}, kl:{}'.format(epoch, elbo.tolist(), kl.tolist(), kl.tolist()))
        # test(model, test_loader)


if __name__ == '__main__':
    main(hp)
