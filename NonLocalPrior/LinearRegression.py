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


def get_condition(Z, U, c, V, d):
    condition1 = Z>-1/c
    condition2 = torch.log(U) < 0.5 * Z**2 + d - d*V + d*torch.log(V)
    condition = condition1 * condition2
    return condition

class MarsagliaTsampler(nn.Module):
    """
    Implement Marsaglia and Tsang’s method as a Gamma variable sampler: https://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/
    """
    def __init__(self, size):
        super().__init__()
        # self.log_gamma_alpha = nn.Parameter(-1.*torch.ones(size))  # TODO: see alpha init matters or not
        self.gamma_alpha = nn.Parameter(2.*torch.ones(size))
        self.size = size

    def forward(self, batch_size):
        self.alpha = torch.relu(self.gamma_alpha) + 1  # right now only for alpha > 1
        d = self.alpha - 1/3
        c = 1. / torch.sqrt(9. * d)
        Z = Normal(0, 1).sample([batch_size, self.size])
        U = Uniform(0, 1).sample([batch_size, self.size])
        V = (1+c*Z)**3

        condition = get_condition(Z, U, c, V, d).type(torch.float)
        out = condition * d*V
        processed_out = torch.stack([out[:,p][out[:,p]>0][:10] for p in range(self.size)], dim=0).t()
        # out = out[out>0]
        detached_gamma_alpha = self.alpha  #.detach()
        return processed_out, detached_gamma_alpha


class SpikeAndSlabSampler(nn.Module):
    """
    Add spike and slab to MarsagliaTsampler
    """
    def __init__(self, p, alternative_sampler=MarsagliaTsampler):
        super(SpikeAndSlabSampler, self).__init__()
        self.p = p
        self.logalpha = nn.Parameter(torch.ones(p))
        self.alternative_sampler = alternative_sampler(size=p)

        # L0 related parameters
        self.zeta = 1.1
        self.gamma = -0.1
        self.beta = 2 / 3
        self.gamma_zeta_logratio = -self.gamma / self.zeta

    def forward(self, batch_size):
        # sample theta
        theta, detached_gamma_alpha = self.alternative_sampler(batch_size=batch_size)  # shape=[batch_size, p]
        z_mean = torch.sigmoid(self.logalpha - self.beta * self.gamma_zeta_logratio)

        # sample z
        u = Uniform(0, 1).sample([10, self.p])  # TODO: 10 is the number of effective samples
        s = torch.sigmoid((torch.log(u / (1 - u)) + self.logalpha) / self.beta)
        z = torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)

        if not self.training:
            z = torch.clamp(torch.sigmoid(self.logalpha) * (self.zeta - self.gamma) + self.gamma, 0, 1).expand_as(u)
            theta = detached_gamma_alpha.expand_as(u)

        return z, z_mean, theta, detached_gamma_alpha, self.logalpha


class LinearModel(nn.Module):
    """
    Wrap around SpikeAndSlabSampler for a linear regression model, use Gamma distribution as variational distribution

    compare with: https://www.tandfonline.com/doi/pdf/10.1080/01621459.2015.1130634?needAccess=true
    """
    def __init__(self, p):
        super(LinearModel, self).__init__()
        self.sampler = SpikeAndSlabSampler(p=p, alternative_sampler=MarsagliaTsampler)
        self.mixweight_logalpha = nn.Parameter(torch.ones(p))  # weight parameter for mixture of 2-component gamma

    def forward(self, x):
        self.z, self.z_mean, self.theta, self.detached_gamma_alpha, self.logalpha = self.sampler(batch_size=64)  # 64 choose 10

        # sample mixture weight using concrete
        u = Uniform(0, 1).sample(self.theta.size())
        weight = torch.sigmoid((torch.log(u / (1 - u)) + self.mixweight_logalpha.expand_as(u)) / (0.9))
        if not self.training:
            weight = torch.exp(self.mixweight_logalpha.expand_as(u)) / (1 + torch.exp(self.mixweight_logalpha.expand_as(u)))

        # sign = 2 * Bernoulli(0.5).sample(self.theta.size()) - 1
        self.signed_theta = weight * self.theta + (1-weight) * (-self.theta)
        self.Theta = self.signed_theta * self.z
        out = x.matmul(self.Theta.mean(dim=0))  # TODO: defer taking mean to the output
        return out

    def kl(self, phi):
        qz = self.z_mean.expand_as(self.theta)
        qlogp = torch.log(nlp_pdf(self.signed_theta, phi, tau=0.358)+1e-8)
        gamma = Gamma(concentration=self.detached_gamma_alpha, rate=1.)
        qlogq = np.log(0.5) + gamma.log_prob(self.theta)  # use unsigned self.theta to compute qlogq
        kl_beta = qlogq - qlogp
        p = 5e-3
        kl_z = qz*torch.log(qz/p) + (1-qz)*torch.log((1-qz)/(1-p))
        kl = (kl_z + qz*kl_beta).sum(dim=1).mean()

        return kl



# class LinearDiracDelta(nn.Module):
#     """
#     Wrap around SpikeAndSlabSampler for a linear regression model, use Dirac delta mass as variational distribution
#     """
#     def __init__(self, p):
#         super(LinearDiracDelta, self).__init__()
#         self.p = p
#         self.theta = nn.Parameter(torch.ones(p))
#         self.logalpha = nn.Parameter(torch.ones(p))
#
#         # L0 related parameters
#         self.zeta = 1.1
#         self.gamma = -0.1
#         self.beta = 2 / 3
#         self.gamma_zeta_logratio = -self.gamma / self.zeta
#
#     def forward(self, x):
#         # sample z
#         u = Uniform(0, 1).sample([10, self.p])  # TODO: 10 is the number of effective samples
#         s = torch.sigmoid((torch.log(u / (1 - u)) + self.logalpha) / self.beta)
#         self.z = torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)
#         self.z_mean = torch.sigmoid(self.logalpha - self.beta * self.gamma_zeta_logratio)
#
#         if not self.training:
#             self.z = torch.clamp(torch.sigmoid(self.logalpha) * (self.zeta - self.gamma) + self.gamma, 0, 1).expand_as(u)
#
#         self.Theta = self.theta * self.z.mean(dim=0)  # TODO: defer taking mean to the output
#         out = x.matmul(self.Theta)
#         return out
#
#     def kl(self, phi):
#         qz = self.z_mean.expand_as(self.theta)
#         qlogp = torch.log(nlp_pdf(self.theta, phi, tau=0.358)+1e-8)
#         qlogq = 0
#         kl_beta = qlogq - qlogp
#         kl_z = qz*torch.log(qz/0.05) + (1-qz)*torch.log((1-qz)/0.95)
#         kl = (kl_z + qz*kl_beta).sum(dim=0)
#         return kl




def loglike(y_hat, y):
    # std = y.std()
    ll = - (y_hat-y)**2/(2*1**2)  # + np.log(1/(np.sqrt(2*np.pi)*1))
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
        kl = linear.kl(phi)
        loss = nll + kl

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        sse = ((y_hat - Y) ** 2).mean().detach().numpy()
        sse_list.append(sse)

        with torch.no_grad():
            linear.eval()
            y_hat = linear(X)
            z = linear.z  # [0, :]
            sse_theta = ((linear.Theta.mean(dim=0) - torch.tensor(truetheta, dtype=torch.float)) ** 2).sum()
            sse_test = ((y_hat - Y) ** 2).mean().detach().numpy()
            sse_theta_list.append(((linear.Theta.mean(dim=0) - torch.tensor(truetheta, dtype=torch.float)) ** 2).sum())

        # print intermediet results
        if i % 50 == 0:
            # print('est.z train mode',linear.z.mean(dim=0).detach().numpy().round(2))
            # z = torch.clamp(torch.sigmoid(linear.logalpha) * (1.2) - 0.1, 0, 1)


            print('\n', 'last 5 responses:', y_hat[-5:].round().tolist(), Y[-5:].round().tolist())
            print('sse_theta:{}, min_sse_theta:{}'.format(sse_theta, np.asarray(sse_theta_list).min()))
            print('est.Thetas: {}, est z:{}'.format(linear.Theta.mean(dim=0)[-5:].tolist(), z.mean(dim=0).detach().numpy().round(2)))
            print('epoch {}, z min: {}, z mean: {}, non-zero: {}'.format(i, linear.z.min(), linear.z.mean(), linear.z.nonzero().shape))
            print('p={}, phi={}, loss: {}, nll:{}, kl:{}. SSE: {}, sse_test: {}'.format(X.shape[0], phi, nll, loss, kl, sse, sse_test))
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
    'save_model': True,
    'save_model_dir': '/extra/yadongl10/git_project/GammaLearningResult'
}


def main(config):
    n = 100
    for p in [1000]:
        for phi in [1, 4, 8]:
            Y, X, truetheta = generate_data(n, p, phi, rho=0, seed=1234)
            linear = train(Y, X, truetheta, phi, epoch=10000)
            test(Y, X, linear)

            if config['save_model']:
                torch.save(linear.state_dict(), config['save_model_dir']+'linear_p1000.pt')




if __name__=='__main__':
    main(config)

