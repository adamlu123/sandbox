from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
from scipy.misc import logsumexp
# from ContLearn_VAE import NonLocalVAE
from torch.distributions import Uniform
import utils
from utils import log

#
#
# parser = argparse.ArgumentParser(description='VAE MNIST Example')
# parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                     help='input batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                     help='number of epochs to train (default: 10)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                     help='how many batches to wait before logging training status')
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
#
# torch.manual_seed(args.seed)
#
# device = torch.device("cuda" if args.cuda else "cpu")
#
# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('/extra/yadongl10/data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('/extra/yadongl10/data', train=False, transform=transforms.ToTensor()),
#     batch_size=100, shuffle=True, **kwargs)


def normal_log_pdf(x, mu, logvar):
    return -0.5 * logvar - 0.5 * np.log(2*np.pi) - (x - mu)**2/(2*logvar.exp())


def mom_log_pdf(x, phi=1, tau=0.358):  # 0.358
    return ((x**2).clamp(min=1e-10)).log() - np.log(tau) - np.log(np.sqrt(2*np.pi*tau*phi)) - x**2/(2*tau*phi)


def imom_log_pdf(x, phi=1, tau=3):  #0.133
    # x = x.sign() * x.abs().clamp(min=1e-5, max=1e5)
    return 0.5*np.log(tau*phi/np.pi) - (x**2).clamp(min=1e-10).log() - (tau*phi) / (x**2).clamp(min=1e-10, max=1e10)



class LearnedHardConcreteSampler(nn.Module):
    """
    Sampler for Hard concrete random variable used for L0 gate
    """
    def __init__(self, p):
        super(LearnedHardConcreteSampler, self).__init__()
        self.p = p
        self.scale = nn.Parameter(-10. * torch.ones(p))

    def forward(self, repeat, logalpha):
        self.beta = 0.9  # F.softplus(self.temparature).clamp(min=0.1, max=0.1)  # 1.1, -0.1, 9/10  #2/3
        scale = (F.softplus(self.scale) + 1).clamp(max=1.1)
        self.zeta, self.gamma = scale[:], - (scale[:] - 1)
        self.gamma_zeta_logratio = torch.log(-self.gamma / self.zeta)

        batch_size = logalpha.shape[0]
        qz = torch.sigmoid(logalpha - self.beta * self.gamma_zeta_logratio)
        u = Uniform(0, 1).sample([repeat, batch_size, self.p]).cuda()
        s = torch.sigmoid((torch.log(u / (1 - u)) + logalpha) / self.beta)
        z = torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)
        if self.training:
            z = torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)
        else:
            z = torch.clamp(torch.sigmoid(logalpha) * (self.zeta - self.gamma) + self.gamma, 0, 1).expand_as(u)
        return z, qz




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
        z = torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)
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
        self.z_sampler = LearnedHardConcreteSampler(p)

    def forward(self, repeat, mu, logvar, logalpha):
        theta, logdet, gaussians = self.alternative_sampler.sample(n=repeat, mu=mu, logvar=logvar)
        logq = normal_log_pdf(gaussians, torch.zeros_like(mu), torch.zeros_like(logvar))
        z, qz = self.z_sampler(repeat, logalpha)  # z.shape = (repeat, batch, p)  qz.shape=(p)
        # logp = normal_log_pdf(theta, torch.zeros_like(mu), torch.zeros_like(logvar))
        # theta = theta.permute([1, 0, 2])  # -> (16, 128, 32)
        out = z * theta
        return out, theta, logq, logdet, z, qz


class NonLocalVAE(nn.Module):
    def __init__(self, latent_dim=20, prior='mom', tau=10, p=0.2):
        super(NonLocalVAE, self).__init__()
        self.alternative_sampler = GaussianAlternative  # GaussianAlternative
        self.fc1 = nn.Linear(784, 400)
        self.fc1_2 = nn.Linear(400, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc23 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc3_2 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 784)

        self.latent_sampler = SpikeAndSlabSampler(p=latent_dim, alternative_sampler=self.alternative_sampler)
        self.prior = prior
        self.tau = tau
        self.p = p

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc1_2(h1))
        return self.fc21(h1), self.fc22(h1), self.fc23(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc3_2(h3))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        self.mu, self.logvar, logalpha = self.encode(x.view(-1, 784))
        z, self.theta, self.logq, self.logdet, self.z, self.qz = self.latent_sampler(1, self.mu, self.logvar, logalpha)
        if isinstance(self.alternative_sampler, FlowAlternative):
            self.theta = self.theta.squeeze()

        self.latent = z
        return self.decode(z.mean(dim=0)), self.mu, self.logvar

    def kl(self):
        p = self.p

        qz = self.qz #.expand_as(self.theta)
        kl_z = qz * torch.log(qz.clamp(min=1e-10) / p) + (1 - qz) * torch.log((1 - qz).clamp(min=1e-10) / (1 - p))

        if self.prior == 'mom':
            qlogp = mom_log_pdf(self.theta, phi=1, tau=self.tau)  # .clamp(min=np.log(1e-10))  .358
        elif self.prior == 'Gaussian':
            qlogp = normal_log_pdf(self.theta, torch.zeros_like(self.theta), torch.zeros_like(self.theta))
        elif self.prior == 'imom':
            qlogp = imom_log_pdf(self.theta, phi=1, tau=self.tau)  # .clamp(min=np.log(1e-10))  .358

        qlogq = self.logq - self.logdet

        kl_beta = qlogq - qlogp  # (16, 128, 32)
        if self.prior == 'Gaussian':
            kl_beta = -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        kl = (kl_z + qz*kl_beta).sum(dim=1)  #.mean(dim=0)  # (128)
        return kl, qlogp.mean(), qlogq.mean()

    def calculate_loss(self, recon_x, x, mode='train'):
        if recon_x.mean() != recon_x.mean():
            stop = 1
        BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

        KLD, qlogp, qlogq = self.kl()  #  (128)
        return BCE + KLD.sum(), BCE



class VAE(nn.Module):
    def __init__(self, latent_dim=200):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc1_2 = nn.Linear(400, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc3_2 = nn.Linear(400, 400)

        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc1_2(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc3_2(h3))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        self.mu, self.logvar = mu, logvar
        z = self.reparameterize(mu, logvar)
        self.z = z
        return self.decode(z), mu, logvar

    def calculate_loss(self, recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        return BCE + KLD, BCE




# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, BCE = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, BCE: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data), BCE.item()/data.shape[0] ))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))



def test_elbo(x, latent_samples = 50):
    for i in range(latent_samples):
        recon_batch, mu, logvar = model(x)
        test_loss_tmp, BCE_tmp = loss_function(recon_batch, x, mu, logvar)
        test_loss_tmp = test_loss_tmp.view(1)
        if i == 0:
            a = test_loss_tmp.cpu().data.numpy()
        else:
            a = np.concatenate((a, test_loss_tmp.cpu().data.numpy()), axis=0)
    likelihood_x = logsumexp(a) - np.log(len(a))
    return likelihood_x


def test(epoch):
    model.eval()
    marginal = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i > 5:
                continue
            data = data.to(device)
            for j in range(data.shape[0]):
                marginal += test_elbo(data[j], latent_samples=50)
            avg_test_marginal = marginal / (data.shape[0] * (i+1))
            print('batch{} avg_test_marginal {}'.format(i, avg_test_marginal))



if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        model = VAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # test(epoch)
        train(epoch)
        test(epoch)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
            # save_image(sample.view(64, 1, 28, 28),
            #            'results/sample_' + str(epoch) + '.png')