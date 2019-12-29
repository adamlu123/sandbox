import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal, Uniform, Gamma, Bernoulli
import matplotlib.pyplot as plt
import numpy as np

def nlp_pdf(x, phi, tau=0.358):
    return x**2*1/tau*(1/np.sqrt(2*np.pi*tau*phi))*torch.exp(-x**2/(2*tau*phi))

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


def lognorm_logpdf(theta, mu, logstd):
    theta = theta.abs()
    logstd = torch.ones_like(logstd)
    return - (theta.log() + logstd + (torch.tensor(2*np.pi)).sqrt().log()) - (theta.log() - mu)**2/(2*(logstd.exp()**2)).clamp(min=1e-1, max=1e1)


def get_qlogq(theta, mu, logstd, weight):
    """
    neg entropy of the variational distribution: qlogq, 1/S*logprob(samples)
    :param theta:
    :return:
    """
    qlogq = torch.where(theta > 0,
                        weight * lognorm_logpdf(theta, mu, logstd),
                        (1 - weight) * lognorm_logpdf(-theta, mu, logstd))
    return qlogq


class LogNormalSampler(nn.Module):
    def __init__(self, p):
        super(LogNormalSampler, self).__init__()
        self.mu = nn.Parameter(.2*torch.ones(p))
        self.logstd = nn.Parameter(1*torch.ones(p))
        self.p = p

    def forward(self, repeat):
        epsilon = Normal(0, 1).sample([repeat, self.p])
        return (self.logstd.exp() * epsilon + self.mu).exp(), self.mu, self.logstd


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

    def forward(self, repeat):
        qz = torch.sigmoid(self.logalpha - self.beta * self.gamma_zeta_logratio)
        u = Uniform(0, 1).sample([repeat, self.p])
        s = torch.sigmoid((torch.log(u / (1 - u)) + self.logalpha) / self.beta)
        if self.training:
            z = torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)
        else:
            z = torch.clamp(torch.sigmoid(self.logalpha) * (self.zeta - self.gamma) + self.gamma, 0, 1).expand_as(u)
        return z, qz


class ConcreteSampler(nn.Module):
    """
    Sampler for concrete relation of Bernoulli variable
    """
    def __init__(self, p, temperature=0.9):
        super(ConcreteSampler, self).__init__()
        self.p = p
        self.temperature = temperature
        self.logalpha = nn.Parameter(torch.ones(p))

    def forward(self, repeat):
        u = Uniform(0, 1).sample([repeat, self.p])
        s = torch.sigmoid((torch.log(u / (1 - u)) + self.logalpha) / self.temperature)
        if not self.training:
            s = self.logalpha.exp() / (1 + self.logalpha.exp())
        return s


class SpikeAndSlabSampler(nn.Module):
    """
    Add spike and slab to LogNormalSampler
    """
    def __init__(self, p, alternative_sampler=LogNormalSampler):
        super(SpikeAndSlabSampler, self).__init__()
        self.p = p
        self.alternative_sampler = alternative_sampler(p)
        self.z_sampler = HardConcreteSampler(p)
        self.mixture_weight_sampler = ConcreteSampler(p)

    def forward(self, repeat):
        theta, mu, logstd = self.alternative_sampler(repeat)
        weight = self.mixture_weight_sampler(repeat)
        z, qz = self.z_sampler(repeat)
        signed_theta = weight * theta + (1 - weight) * (-theta)
        out = z * signed_theta
        return out, signed_theta, mu, logstd, weight, z, qz


class LinearModel(nn.Module):
    """
    Wrap around SpikeAndSlabSampler for a linear regression model, use Gamma distribution as variational distribution
    compare with: https://www.tandfonline.com/doi/pdf/10.1080/01621459.2015.1130634?needAccess=true
    """
    def __init__(self, p, bias=False, alternative_sampler=LogNormalSampler):
        super(LinearModel, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.ones(p[1]))

        if isinstance(p, tuple):
            p = p[0] * p[1]
        self.alternative_sampler = alternative_sampler
        self.sampler = SpikeAndSlabSampler(p=p, alternative_sampler=alternative_sampler)

    def forward(self, x):
        self.Theta, self.signed_theta, self.mu, self.logstd, self.weight, self.z, self.qz = self.sampler(repeat=100)
        out = x.matmul(self.Theta.mean(dim=0).view(x.shape[1], -1))  # TODO: defer taking mean to the output
        return out.squeeze()

    def kl(self, phi):
        p = 5e-3
        qz = self.qz.expand_as(self.Theta)
        kl_z = qz * torch.log(qz / p) + (1 - qz) * torch.log((1 - qz) / (1 - p))
        qlogp = torch.log(nlp_pdf(self.signed_theta, phi, tau=0.358)+1e-8)   # shape=(60, p)
        qlogq = get_qlogq(self.signed_theta, self.mu, self.logstd, self.weight)

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
    optimizer = optim.SGD(linear.parameters(), lr=0.0001, momentum=0.9)
    sse_list = []
    sse_theta_list = []
    for i in range(epoch):
        linear.train()
        optimizer.zero_grad()

        # forward pass and compute loss
        y_hat = linear(X)
        nll = -loglike(y_hat, Y)
        kl, qlogp, qlogq = linear.kl(phi)
        loss = nll + kl
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
        if i % 200 == 0:
            z = linear.z
            print('\n', 'last 5 responses:', y_hat[-5:].round().tolist(), Y[-5:].round().tolist())
            print('sse_theta:{}, min_sse_theta:{}'.format(sse_theta, np.asarray(sse_theta_list).min()))
            print('est.Thetas: {}, est z:{}'.format(linear.Theta.mean(dim=0)[-5:].tolist(), z.mean(dim=0).detach().numpy()))
            print('epoch {}, z min: {}, z mean: {}, non-zero: {}'.format(i, z.min(), z.mean(), z.nonzero().shape))
            # print('p={}, phi={}, loss: {}, nll:{}, kl:{}. SSE: {}, sse_test: {}'.format(X.shape[0], phi, nll, loss, kl, sse, sse_test))
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
    for p in [10]:
        for phi in [1, 4, 8]:
            Y, X, truetheta = generate_data(n, p, phi, rho=0, seed=1234)
            linear = train(Y, X, truetheta, phi, epoch=10000)
            test(Y, X, linear)

            # if config['save_model']:
            #     torch.save(linear.state_dict(), config['save_model_dir']+'lognorm_nlp_linear_p1000.pt')


if __name__=='__main__':
    main(config)

