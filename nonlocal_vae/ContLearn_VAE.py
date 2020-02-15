import argparse
import torch
import torch.nn.functional as F
import numpy as np
import utils
from utils import log

import argparse
import torch.utils.data
from torch import nn, optim
from torch.distributions import Uniform, Normal
from torchvision import datasets, transforms
import pickle as pkl
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = 'cuda' # torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def normal_log_pdf(x, mu, logvar):
    return -0.5 * logvar - 0.5 * np.log(2*np.pi) - (x - mu)**2/(2*logvar.exp())


def mom_log_pdf(x, phi=1, tau=0.358):  # 0.358
    return ((x**2).clamp(min=1e-10)).log() - np.log(tau) - np.log(np.sqrt(2*np.pi*tau*phi)) - x**2/(2*tau*phi)



class HardConcreteSampler(nn.Module):
    """
    Sampler for Hard concrete random variable used for L0 gate
    """
    def __init__(self, p):
        super(HardConcreteSampler, self).__init__()
        self.p = p
        self.zeta, self.gamma, self.beta = 1.1, -0.1, 9/10
        self.gamma_zeta_logratio = -self.gamma / self.zeta
        # self.logalpha.data.uniform_(np.log(0.2), np.log(10))

    def forward(self, repeat, logalpha):
        batch_size = logalpha.shape[0]
        qz = torch.sigmoid(logalpha - self.beta * self.gamma_zeta_logratio)
        u = Uniform(0, 1).sample([repeat, batch_size, self.p]).cuda()
        s = torch.sigmoid((torch.log(u / (1 - u)) + logalpha) / self.beta)
        if self.training:
            z = torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)
        else:
            z = torch.clamp(torch.sigmoid(logalpha) * (self.zeta - self.gamma) + self.gamma, 0, 1).expand_as(u)
        return z, qz


class BaseFlow(nn.Module):
    def sample(self, n=1, context=None, **kwargs):
        dim = self.p
        if isinstance(self.p, int):
            dim = [dim, ]

        if not kwargs:
            spl = Variable(torch.FloatTensor(n, *dim).normal_())
            lgd = torch.zeros(n, *dim)

        else:
            mu = kwargs['mu']
            logvar = kwargs['logvar']
            batch_size = mu.shape[0]
            spl = Variable(torch.FloatTensor(n, batch_size, *dim).normal_()).cuda()
            std = torch.exp(0.5 * logvar)
            spl = mu + std * spl
            lgd = torch.zeros(n, batch_size, *dim).cuda()

        if hasattr(self, 'gpu'):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.gpu()

        return self.forward((spl, lgd, context))

    def cuda(self):
        self.gpu = True
        return super(BaseFlow, self).cuda()


class SigmoidFlow(BaseFlow):
    def __init__(self, num_ds_dim=4):
        super(SigmoidFlow, self).__init__()
        self.num_ds_dim = num_ds_dim
        self.act_a = lambda x: utils.softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: utils.softmax(x, dim=2)

    def forward(self, x, logdet, dsparams, mollify=0.0, delta=utils.delta):
        ndim = self.num_ds_dim
        a_ = self.act_a(dsparams[:, :, 0 * ndim:1 * ndim])
        b_ = self.act_b(dsparams[:, :, 1 * ndim:2 * ndim])
        w = self.act_w(dsparams[:, :, 2 * ndim:3 * ndim])

        a = a_ * (1 - mollify) + 1.0 * mollify
        b = b_ * (1 - mollify) + 0.0 * mollify
        pre_sigm = a * x[:, :, :, None] + b   # a.shape=(16, 32, 4)
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm, dim=3)
        x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)
        xnew = x_

        logj = F.log_softmax(dsparams[:, :, 2 * ndim:3 * ndim], dim=2) + \
               utils.logsigmoid(pre_sigm) + \
               utils.logsigmoid(-pre_sigm) + log(a)

        logj = utils.log_sum_exp(logj, 3).sum(3)
        logdet_ = logj + np.log(1 - delta) - (log(x_pre_clipped) + log(-x_pre_clipped + 1))
        logdet = logdet_.permute([1,0,2]) + logdet
        return xnew, logdet


class FlowAlternative(BaseFlow):
    """
    Independent component-wise flow using SigmoidFlow
    """
    def __init__(self, p, num_ds_dim=4, num_ds_layers=2):
        super(FlowAlternative, self).__init__()
        self.p = p
        self.num_ds_dim = num_ds_dim
        self.num_ds_layers = num_ds_layers
        self.nparams = self.num_ds_dim * 3
        self.dsparams = nn.Parameter(torch.ones((p, num_ds_layers * self.nparams)))
        self.dsparams.data.uniform_(-0.001, 0.001)  # TODO check whether it init correctly
        self.sf = SigmoidFlow(num_ds_dim)

    def forward(self, inputs):
        x, logdet, context = inputs
        repeat = x.shape[0]

        h = x.permute([1,0,2])  # x.shape=(16, 128, 32) -> (128, 16, 32) # x.view(x.size(0), -1)   # TODO: ?
        for i in range(self.num_ds_layers):
            params = self.dsparams[:, i * self.nparams:(i + 1) * self.nparams]   # shape=(p, nparams)
            params = params.unsqueeze(0).repeat(repeat, 1, 1)  # shape=(repeat, p, nparams)
            h, logdet = self.sf(h, logdet, params, mollify=0.0)

        return h, logdet, x


class GaussianAlternative(nn.Module):
    """
    Independent Gaussian alternative distribution
    """
    def __init__(self, p):
        super(GaussianAlternative, self).__init__()
        self.p = p

    def forward(self, inputs, mu, logvar):
        return inputs * (0.5 * logvar).exp() + mu

    def sample(self, n, mu, logvar):
        noise = torch.rand(n, mu.shape[0], self.p).cuda() #  rep, batch, latent_dim
        x = self.forward(noise, mu, logvar)
        logdet = 0.5*logvar
        return x, logdet, noise


class SpikeAndSlabSampler(nn.Module):
    """
    Spike and slab sampler with Dirac spike and FlowAlternative slab.
    """
    def __init__(self, p, alternative_sampler=GaussianAlternative):  # FlowAlternative,  GaussianAlternative
        super(SpikeAndSlabSampler, self).__init__()
        self.p = p
        self.alternative_sampler = alternative_sampler(p)
        self.z_sampler = HardConcreteSampler(p)

    def forward(self, repeat, mu, logvar, logalpha):
        theta, logdet, gaussians = self.alternative_sampler.sample(n=repeat, mu=mu, logvar=logvar)
        logq = normal_log_pdf(gaussians, torch.zeros_like(mu), torch.zeros_like(logvar))
        z, qz = self.z_sampler(repeat, logalpha)  # z.shape = (repeat, batch, p)  qz.shape=(p)
        # theta = theta.permute([1, 0, 2])  # -> (16, 128, 32)
        out = z * theta
        return out, theta, logq, logdet, z, qz

    # def reparameterize(self, n=repeat, mu=mu, logvar=logvar):
    #     noise = torch.rand(n, mu.shape[0], self.p).cuda()
    #     return noise * (0.5 * logvar).exp() + mu,

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(784, 400)
        self.fc_output = nn.Linear(400, latent_dim*3)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        out = self.fc_output(x)
        return out.chunk(3, 1)
        # return out[:, :32], out[:, 32:64], out[:, 64:96]


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(input_dim, 400)
        self.fc4 = nn.Linear(400, output_dim)

    def forward(self, z):
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        return torch.sigmoid(z)


class Encoder_rat(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder_rat, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc_output = nn.Linear(20, latent_dim*3)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc_output(x)
        return out.chunk(3, 1)


class Decoder_rat(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder_rat, self).__init__()
        self.fc3 = nn.Linear(input_dim, 20)
        self.fc4 = nn.Linear(20, 40)
        self.fc5 = nn.Linear(40, output_dim)

    def forward(self, z):
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(z))


class NonLocalVAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32, prior='mom', tau=10):
        super(NonLocalVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.latent_sampler = SpikeAndSlabSampler(p=latent_dim)
        self.prior = prior
        self.tau = tau

    def forward(self, data):
        mu, logvar, logalpha = self.encoder(data)
        latent, self.theta, self.logq, self.logdet, self.z, self.qz = self.latent_sampler(1, mu, logvar, logalpha)
        recon_batch = self.decoder(latent)
        return recon_batch

    def kl(self):
        p = 0.9
        qz = self.qz.expand_as(self.theta)
        kl_z = qz * torch.log(qz.clamp(min=1e-10) / p) + (1 - qz) * torch.log((1 - qz).clamp(1e-10) / (1 - p))
        if self.prior == 'mom':
            qlogp = mom_log_pdf(self.theta, phi=1, tau=self.tau)  # .clamp(min=np.log(1e-10))  .358
        elif self.prior == 'Gaussian':
            qlogp = normal_log_pdf(self.theta, torch.zeros_like(self.theta), torch.zeros_like(self.theta))
        qlogq = self.logq - self.logdet

        kl_beta = qlogq - qlogp  # (16, 128, 32)
        kl = (kl_z + qz*kl_beta).sum(dim=2).mean(dim=0)  # (128)
        return kl.sum(), qlogp.mean(), qlogq.mean()

    def calculate_loss(self, recon_x, x, mode='train'):
        BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

        KLD, _, _ = self.kl()
        if mode == 'test':
            return BCE + KLD, F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='none').sum(dim=1)
        else:
            return BCE + KLD, BCE

def train(epoch, model, optimizer, train_loader, classes):
    train_loss = 0
    train_loss_list = []
    for batch_idx, data in enumerate(train_loader):
        model.train()
        data = data.to(device)
        optimizer.zero_grad()

        # forward pass
        recon_batch = model(data).mean(dim=0)   # (16, 128, 784) -> (128, 784)

        # compute elbo loss and print intermediate results
        nll = F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction='none')     # (128, 784)
        nll = nll.sum(1)    # (128)
        kl, qlogp, qlogq = model.kl(phi=1, classes=classes)
        loss = nll + kl
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()/data.shape[0] ))
        train_loss_list.append(loss.item())

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss_list



def kNNClassifer(latent_train, latent_test, k):
    num_each_class = latent_train.shape[0]/10
    latent_test = latent_test.unsqueeze(1).repeat(1, latent_train.shape[0], 1)
    distance = ((latent_test - latent_train) ** 2).sum(dim=2)  # shape=(num_sample, num_base_sample)
    _, index = torch.topk(distance, k, dim=1)  # index shape=(num_sample, k)
    index = index // num_each_class
    pred, _ = torch.mode(index, dim=1)
    return pred


def get_latent(data):
    mu, logvar, logalpha = model.encoder(data.float())
    latent, _, _, _, _, _ = model.latent_sampler(1, mu, logvar, logalpha)
    return latent.squeeze()


def test(model, k, num_each_class):
    model.eval()  # TODO: notice there is a difference in L0 gate
    base, test, label = utils.get_test_data(num_each_class=num_each_class)
    latent_test = get_latent(test)
    latent_base = get_latent(base)

    pred = kNNClassifer(latent_base, latent_test, k)

    acc = utils.get_accuracy(pred, label).tolist()
    print('test acc of {}-NN is {}:'.format(k, acc))


if __name__ == "__main__":
    model = NonLocalVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model.parameters)

    train_loss_list = []
    for i in range(10):
        print('training on class {}'.format(i))
        train_loader = utils.get_train_loader(subset=i, batch_size=128)
        for epoch in range(1, args.epochs + 1):
            train_loss_list += train(epoch, model, optimizer, train_loader, classes=i)

        test(model, k=10, num_each_class=100)
    with open('vae_bayes_loss.pkl'.format(epoch), 'wb') as f:
        pkl.dump(train_loss_list, f)

