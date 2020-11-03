

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
    'epochs': 10000
}


def log_prob_beta(theta, eta, value):
    return (theta-1)*torch.log(value) + (eta-1)*torch.log(1-value) + \
           torch.lgamma(theta+eta) - torch.lgamma(theta) -torch.lgamma(eta)


# def log_prob_gamma(alpha, beta, value):
#     return (self.concentration * torch.log(self.rate) +
#                 (self.concentration - 1) * torch.log(value) -
#                 self.rate * value - torch.lgamma(self.concentration))

class StochasticGammaLayer(Function):
    def __init__(self):
        super(StochasticGammaLayer, self).__init__()

    @staticmethod
    def forward(ctx, alpha, beta):
        w = Gamma(concentration=alpha, rate=beta).sample()
        ctx.save_for_backward(w, alpha, beta)
        return w

    @staticmethod
    def backward(ctx, output_grad):
        w, alpha, beta = ctx.saved_tensors
        log_w = w.log()
        log_w = log_w + output_grad*w * 1  # do the update on the log scale and transform it back
        updated_w = log_w.exp()
        updated_w = updated_w.detach()
        ll = Gamma(concentration=alpha, rate=beta).log_prob(updated_w)
        ll.backward()
        return alpha.grad, beta.grad


class StochasticBetaLayer(Function):
    def __init__(self):
        super(StochasticGammaLayer, self).__init__()

    @staticmethod
    def forward(ctx, theta, eta):
        z = Beta(concentration0=theta, concentration1=eta).sample()
        ctx.save_for_backward(z, theta, eta)
        return z

    @staticmethod
    def backward(ctx, output_grad):
        import pydevd
        pydevd.settrace(suspend=False, trace_only_current_thread=True)
        z, theta, eta = ctx.saved_tensors
        logit_z = torch.log(z / (1 - z))
        logit_z = logit_z + logit_z_grad * 1  # do gradient update on logit scale, then transform it back.
        updated_z = torch.sigmoid(logit_z)
        updated_z = updated_z.detach()
        ll = log_prob_beta(theta, eta, updated_z)  # TODO: check why require_grad becomes False?!
        test = theta + eta
        ll.backward()
        return theta.grad, eta.grad


class BetaGammaMF(nn.Module):
    def __init__(self, StochasticGammaLayer, StochasticBetaLayer):
        super(BetaGammaMF, self).__init__()
        # self.prior = prior
        self.StochasticGammaLayer = StochasticGammaLayer
        self.StochasticBetaLayer = StochasticBetaLayer
        self.logalpha = nn.Parameter(torch.ones(100, 784))
        self.logbeta = nn.Parameter(torch.ones(100, 784))
        self.logtheta = nn.Parameter(torch.ones(5000, 100))
        self.logeta = nn.Parameter(torch.ones(5000, 100))

    def forward(self, x):
        alpha, beta, theta, eta = self.logalpha.exp(), self.logbeta.exp(), \
                                  self.logtheta.exp(), self.logeta.exp()
        w = self.StochasticGammaLayer.apply(alpha, beta)
        z = self.StochasticBetaLayer.apply(theta, eta)
        logit_z = torch.log(z / (1 - z))
        out = torch.sigmoid(logit_z.matmul(w))
        return out

    def get_kl(self):
        gamma_q = Gamma(concentration=self.logalpha.exp(), rate=self.logbeta.exp())
        gamma_p = Gamma(0.1*torch.ones_like(self.logalpha), 0.3*torch.ones_like(self.logalpha))
        beta_q = Beta(self.logtheta.exp(), self.logeta.exp())
        beta_p = Beta(torch.ones_like(self.logtheta), torch.ones_like(self.logtheta))
        # kl = _kl_beta_beta(beta_q, beta_p) + _kl_gamma_gamma(gamma_q, gamma_p)
        kl = kl_divergence(beta_q, beta_p).sum() + kl_divergence(gamma_q, gamma_p).sum()
        return kl

    def get_elbo(self, out, data):
        ll = data * out.log() + (1 - data) * (1 - out + 1e-10).log()
        elbo = ll.sum() - self.get_kl()
        return elbo


def train(model, optimizer, train_loader):
    model.train()
    elbo_list = []
    for batch_idx, data in enumerate(train_loader):
        data = data[0]
        optimizer.zero_grad()

        out = model(data)
        elbo = model.get_elbo(out, data)
        loss = - elbo
        loss.backward()
        optimizer.step()

        elbo_list.append(elbo)
        if batch_idx % hp['log_interval'] == 0:
            print(elbo.tolist())


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


def main(hp):
    # Define model
    print('start define model')
    model = BetaGammaMF(StochasticGammaLayer, StochasticBetaLayer).to(hp['device'])

    # prepare data
    print('start loading data')
    train_loader = prepare_data('train', hp)
    test_loader = prepare_data('test', hp)

    # define opt
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training
    print('start training')
    for epoch in range(hp['epochs']):
        print('Epoch: ', epoch)
        train(model, optimizer, train_loader)
        # test(model, test_loader)

if __name__ == '__main__':
    main(hp)
