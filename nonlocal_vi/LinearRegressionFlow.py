import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import log
from torch import optim
from torch.distributions import Normal, Uniform, Gamma, Bernoulli
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import utils
torch.manual_seed(1234)


def logit(x):
    return torch.log(x/(1-x))


def nlp_pdf(x, phi, tau=0.358):
    return x**2*1/tau*(1/np.sqrt(2*np.pi*tau*phi))*torch.exp(-x**2/(2*tau*phi))


def nlp_log_pdf(x, phi, tau=0.358):
    return (x**2).clamp(min=1e-2).log() - np.log(tau) - np.log(np.sqrt(2*np.pi*tau*phi)) - x**2/(2*tau*phi)


def generate_data(n, p, phi, rho, seed):
    np.random.seed(seed)
    noise = np.random.normal(loc=0, scale=np.sqrt(phi), size=[n])  # noise, with std phi

    sigma = rho * np.ones((p, p))
    np.fill_diagonal(sigma, 1)
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=n)  # X, with correlation rho

    truetheta = np.asarray([0.6, 1.2, 1.8, 2.4, 3])
    theta = np.zeros(p)
    theta[-5:] = truetheta  # beta

    Y = np.matmul(X, theta) + noise
    return Y, X, theta


class HardConcreteSampler(nn.Module):
    """
    Sampler for Hard concrete random variable used for L0 gate
    """
    def __init__(self, p):
        super(HardConcreteSampler, self).__init__()
        self.p = p
        self.zeta, self.gamma, self.beta = 1.1, -0.1, 2/3
        self.gamma_zeta_logratio = -self.gamma / self.zeta
        self.logalpha = nn.Parameter(torch.ones(p))
        # self.logalpha.data.uniform_(np.log(0.2), np.log(10))

    def forward(self, repeat):
        qz = torch.sigmoid(self.logalpha - self.beta * self.gamma_zeta_logratio)
        u = Uniform(0, 1).sample([repeat, self.p])
        s = torch.sigmoid((torch.log(u / (1 - u)) + self.logalpha) / self.beta)
        if self.training:
            z = torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)
        else:
            z = torch.clamp(torch.sigmoid(self.logalpha) * (self.zeta - self.gamma) + self.gamma, 0, 1).expand_as(u)
        return z, qz


class BaseFlow(nn.Module):
    def sample(self, n=1, context=None, **kwargs):
        dim = self.p
        if isinstance(self.p, int):
            dim = [dim, ]

        spl = Variable(torch.FloatTensor(n, *dim).normal_())
        # lgd = Variable(torch.from_numpy(
        #     np.zeros(n, self.p).astype('float32')))
        lgd = torch.zeros(n, *dim)
        # if context is None:
        #     context = Variable(torch.from_numpy(
        #         np.ones((n, self.context_dim)).astype('float32')))

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

        pre_sigm = a * x[:, :, None] + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm, dim=2)
        x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)
        xnew = x_

        logj = F.log_softmax(dsparams[:, :, 2 * ndim:3 * ndim], dim=2) + \
               utils.logsigmoid(pre_sigm) + \
               utils.logsigmoid(-pre_sigm) + log(a)

        logj = utils.log_sum_exp(logj, 2).sum(2)
        logdet_ = logj + np.log(1 - delta) - (log(x_pre_clipped) + log(-x_pre_clipped + 1))
        logdet = logdet_ + logdet

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

        h = x  # x.view(x.size(0), -1)   # TODO: ?
        for i in range(self.num_ds_layers):
            params = self.dsparams[:, i * self.nparams:(i + 1) * self.nparams]   # shape=(p, nparams)
            params = params.unsqueeze(0).repeat(repeat, 1, 1)  # shape=(repeat, p, nparams)
            h, logdet = self.sf(h, logdet, params, mollify=0.0)

        return h, logdet


# class LocalLinear(nn.Module):
#     def __init__(self,in_features,local_features,kernel_size,padding=0,stride=1,bias=True,softmax_weight=False):
#         super(LocalLinear, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#
#         fold_num = (in_features+2*padding-self.kernel_size)//self.stride+1
#         self.weight = nn.Parameter(torch.randn(fold_num,kernel_size,local_features))
#         if softmax_weight:
#             self.weight = F.softmax(self.weight)   # TODO: check which dim should apply softmax
#         self.bias = nn.Parameter(torch.randn(fold_num,local_features)) if bias else None
#
#     def forward(self, x):
#         x = F.pad(x,[self.padding]*2,value=0)
#         x = x.unfold(-1,size=self.kernel_size,step=self.stride)
#         x = torch.matmul(x.unsqueeze(2),self.weight).squeeze(2)+self.bias
#         return x
#
#
# class FlowAlternative(nn.Module):
#     """
#     Independent component-wise flow
#     """
#     def __init__(self, p):
#         super(FlowAlternative, self).__init__()
#         self.p = p
#         self.LocalLinear_in = LocalLinear(p, local_features=5, kernel_size=1)
#         self.LocalLinear_out = LocalLinear(p, local_features=1, kernel_size=1)
#
#     def forward(self, repeat):
#         epsilon = Normal(0, 1).sample([repeat, self.p])
#         x = torch.sigmoid(self.LocalLinear_in(epsilon))  # shape=(d,p)
#         out = logit(self.LocalLinear_out(x))
#         return out


class SpikeAndSlabSampler(nn.Module):
    """
    Spike and slab sampler with Dirac spike and FlowAlternative slab.
    """
    def __init__(self, p, alternative_sampler=FlowAlternative):
        super(SpikeAndSlabSampler, self).__init__()
        self.p = p
        self.alternative_sampler = alternative_sampler(p)
        self.z_sampler = HardConcreteSampler(p)

    def forward(self, repeat):
        theta, logdet = self.alternative_sampler.sample(n=repeat)
        z, qz = self.z_sampler(repeat)
        out = z * theta
        return out, theta, logdet, z, qz


class LinearModel(nn.Module):
    """
    Wrap around SpikeAndSlabSampler for a linear regression model.
    compare with: https://www.tandfonline.com/doi/pdf/10.1080/01621459.2015.1130634?needAccess=true
    """
    def __init__(self, p, bias=False, alternative_sampler=FlowAlternative):
        super(LinearModel, self).__init__()
        self.add_bias = bias
        if self.add_bias:
            self.bias = nn.Parameter(torch.ones(p[1]))
            self.bias.data.uniform_(-0.1, 0.1)

        if isinstance(p, tuple):
            p = p[0] * p[1]
        self.alternative_sampler = alternative_sampler
        self.sampler = SpikeAndSlabSampler(p=p, alternative_sampler=alternative_sampler)

    def forward(self, x):
        self.Theta, self.theta, self.logdet, self.z, self.qz = self.sampler(repeat=32)
        if self.add_bias:
            out = x.matmul(self.Theta.mean(dim=0).view(x.shape[1], -1)) + self.bias
        else:
            out = x.matmul(self.Theta.mean(dim=0).view(x.shape[1], -1))  # TODO: defer taking mean to the output
        return out.squeeze()

    def kl(self, phi):
        p = 1e-2
        qz = self.qz.expand_as(self.Theta)
        kl_z = qz * torch.log(qz / p) + (1 - qz) * torch.log((1 - qz) / (1 - p))
        qlogp = nlp_log_pdf(self.theta, phi, tau=0.358).clamp(min=np.log(1e-10))
        qlogq = -self.logdet

        kl_beta = qlogq - qlogp
        kl = (kl_z + qz*kl_beta).sum(dim=1).mean()
        return kl, qlogp.mean(), qlogq.mean()



def loglike(y_hat, y):
    ll = - (y_hat-y)**2/(2*1**2)
    return ll.sum()


def train(Y, X, truetheta, phi, epoch=10000):
    Y = torch.tensor(Y, dtype=torch.float)
    X = torch.tensor(X, dtype=torch.float)
    linear = LinearModel(p=X.shape[1])
    # optimizer = optim.SGD(linear.parameters(), lr=0.0001, momentum=0.9)
    optimizer = optim.Adam(linear.parameters(), lr=0.01, weight_decay=0)
    sse_list = []
    sse_theta_list = []
    for i in range(epoch):
        linear.train()
        optimizer.zero_grad()

        # forward pass and compute loss
        y_hat = linear(X)
        nll = -loglike(y_hat, Y)
        kl, qlogp, qlogq = linear.kl(phi)
        loss = nll + 1 * kl
        # print('qlogp, qlogq', qlogp.data, qlogq.data)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        sse = ((y_hat - Y) ** 2).mean().detach().numpy()
        sse_list.append(sse)

        with torch.no_grad():
            linear.eval()
            y_hat = linear(X)
            z = linear.z
            sse_theta = ((linear.Theta.mean(dim=0) - torch.tensor(truetheta, dtype=torch.float)) ** 2).sum()
            sse_test = ((y_hat - Y) ** 2).mean().detach().numpy()
            sse_theta_list.append(((linear.Theta.mean(dim=0) - torch.tensor(truetheta, dtype=torch.float)) ** 2).sum())

        # print intermediet results
        if i % 50 == 0:
            z = linear.z.mean(dim=0).cpu().numpy()
            print('\n', i, 'last 5 responses:', y_hat[-5:].round().tolist(), Y[-5:].round().tolist())
            print('sse_theta:{}, min_sse_theta:{}'.format(sse_theta, np.asarray(sse_theta_list).min()))
            print('est.thetas: {}, est z:{}'.format(linear.Theta.mean(dim=0)[-5:].tolist(), z[-5:]))
            print('epoch {}, z min: {}, z mean: {}, non-zero: {}'.format(i, z.min(), z.mean(), z.nonzero()[0].shape))
            print('nll, kl', nll.tolist(), kl.tolist())
            threshold = utils.search_threshold(z, 0.05)
            print('threshold', threshold)
            print('number of cov above threshold', np.sum(z>threshold))
            # print('p={}, phi={}, loss: {}, nll:{}, kl:{}. SSE: {}, sse_test: {}'.format(X.shape[0], phi, nll, loss, kl, sse, sse_test))
        if i % 100 == 0:
            utils.plot(linear.Theta, savefig=True)

    plt.plot(sse_list)
    # plt.savefig('/extra/yadongl10/git_project/GammaLearningResult/sse.png', dpi=100)
    return linear


def test(Y, X, model):
    model.eval()
    y_hat = model(X)
    sse = ((y_hat - Y) ** 2).mean().detach().numpy()

    print('test SSE:{}'.format(sse))
    if model.z.nonzero().shape[0] < 10:
        print(model.z.nonzero(), model.z[-5:].tolist())


config = {
    'save_model': False,
    'save_model_dir': '/extra/yadongl10/git_project/nlpresult'
}


def main(config):
    n = 100
    for p in [100]:
        for phi in [1, 4, 8]:
            Y, X, truetheta = generate_data(n, p, phi, rho=0, seed=1234)
            linear = train(Y, X, truetheta, phi, epoch=10000)  # 10000
            test(Y, X, linear)

            # if config['save_model']:
            #     torch.save(linear.state_dict(), config['save_model_dir']+'lognorm_nlp_linear_p1000.pt')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Flow spike and slab')
    parser.add_argument(
        '--p',
        type=int,
        default=100,
        help='number of covariates (default: 100)')
    main(config)