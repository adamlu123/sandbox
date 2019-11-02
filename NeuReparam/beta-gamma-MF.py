import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Function
import pickle as pkl
from torch.distributions import Gamma, Beta
import torch.distributions.kl.kl_divergence as kl_divergence


hp = {
    'data_folder': '/extra/yadongl10/data/MNIST/',
    'device': 'cuda',
    'epochs': 10000
}


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
        updated_w = w + output_grad * 1
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
        z, theta, eta = ctx.saved_tensors
        updated_z = z + output_grad * 1
        updated_z = updated_z.detach()
        ll = Beta(concentration0=theta, concentration1=eta).log_prob(updated_z)
        ll.backward()
        return theta.grad, eta.grad


class BetaGammaMF(nn.Module):
    def __init__(self, StochasticGammaLayer, StochasticBetaLayer):
        super(BetaGammaMF, self).__init__()
        # self.prior = prior
        self.StochasticGammaLayer = StochasticGammaLayer
        self.StochasticBetaLayer = StochasticBetaLayer
        self.logalpha = nn.Parameter(torch.ones(100, 1024))
        self.logbeta = nn.Parameter(torch.ones(100, 1024))
        self.logtheta = nn.Parameter(torch.ones(500, 100))
        self.logeta = nn.Parameter(torch.ones(500, 100))

    def forward(self, x):
        alpha, beta, theta, eta = self.logalpha.exp(), self.logbeta.exp(), \
                                  self.logtheta.exp(), self.logeta.exp()
        w = self.StochasticGammaLayer(alpha, beta)
        z = self.StochasticBetaLayer(theta, eta)
        logit_z = torch.log(z / (1 - z))
        out = torch.sigmoid(logit_z.matmul(w))
        return out

    def get_kl(self):
        gamma_q = Gamma(concentration=self.logalpha.exp(), rate=self.logbeta.exp())
        gamma_p = Gamma(0.1, 0.3)
        beta_q = Beta(self.logtheta.exp(), self.logeta.exp())
        beta_p = Beta(1, 1)
        # kl = _kl_beta_beta(beta_q, beta_p) + _kl_gamma_gamma(gamma_q, gamma_p)
        kl = kl_divergence(beta_q, beta_p) + kl_divergence(gamma_q, gamma_p)
        return kl

    def get_elbo(self, out, data):
        ll = data * out.log() + (1 - data) * (1 - out).log()
        elbo = ll - self.get_kl()
        return elbo


def train(model, optimizer, train_loader):
    model.train()
    elbo_list = []
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()

        out = model(data)
        elbo = model.get_elbo(out)
        loss = - elbo
        loss.backward()
        optimizer.step()

        elbo_list.append(elbo)
        if batch_idx % hp['log_interval'] == 0:
            print(elbo.tolist())


# def test(model, test_loader):


def prepare_data(subset, hp):
    if subset == 'train':
        with open(hp['data_folder'] + 'binarized_mnist_train.amat', 'rb') as f:
            data = pkl.load(f)
            data = data[:5000, :, :]
            print(data.shape)
    elif subset == 'test':
        with open(hp['data_folder'] + 'binarized_mnist_test.amat', 'rb') as f:
            data = pkl.load(f)
            data = data[:1000, :, :]
            print(data.shape)

    data = torch.tensor(data, dtype=torch.float32).to(hp['device'])
    data = torch.utils.data.TensorDataset(data)
    data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
    return data_loader


def main(hp):
    # Define model
    model = BetaGammaMF(StochasticGammaLayer, StochasticBetaLayer)

    # prepare data
    train_loader = prepare_data('train', hp)
    test_loader = prepare_data('test', hp)

    # define opt
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training
    for epoch in range(hp['epochs']):
        print('Epoch: ', epoch)
        train(model, optimizer, train_loader)
        # test(model, test_loader)
