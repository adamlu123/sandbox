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
import pickle as pkl

torch.manual_seed(123)
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="3"


def logit(x):
    return torch.log(x/(1-x))


def normal_log_pdf(x, mu, logvar):
    return -0.5 * logvar - 0.5 * np.log(2*np.pi) - (x - mu)**2/(2*logvar.exp())


def nlp_pdf(x, phi, tau=0.358):
    return x**2*1/tau*(1/np.sqrt(2*np.pi*tau*phi))*torch.exp(-x**2/(2*tau*phi))


def mom_log_pdf(x, phi=1, tau=0.358):  # 0.358
    return ((x**2).clamp(min=1e-10)).log() - np.log(tau) - np.log(np.sqrt(2*np.pi*tau*phi)) - x**2/(2*tau*phi)


def imom_log_pdf(x, phi=1, tau=3):  #0.133
    # x = x.sign() * x.abs().clamp(min=1e-5, max=1e5)
    return 0.5*np.log(tau*phi/np.pi) - (x**2).clamp(min=1e-10).log() - (tau*phi) / (x**2).clamp(min=1e-10, max=1e10)


def pemom_log_pdf(x, phi=1, tau=0.358):
    return (np.sqrt(2) - (tau*phi) / (x**2).clamp(min=1e-10)) - np.log(np.sqrt(2*np.pi*tau*phi)) - x**2/(2*tau*phi)


def normal_log_prob(x, var=10):
    return np.log(1/np.sqrt(2*np.pi)) - 0.5 * np.log(var) - x**2/(2*var)


def generate_data(n, p, phi, rho, seed, load_data=True):
    truetheta = np.asarray([1, 2, 3, 4, 5])  # [0.6, 1.2, 1.8, 2.4, 3], [6, 7, 8, 9, 10]
    if load_data:
        truetheta = np.asarray([0.6, 1.2, 1.8, 2.4, 3]*2)
        theta = np.zeros(p)
        theta[-truetheta.shape[0]:] = truetheta  # beta
        data_dir = '/extra/yadongl10/data/non_local_simulation/sim_data_theta_5'
        Y = np.loadtxt(data_dir + '/y_p{}_n{}_rho{}_phi{}.txt'.format(p, n, rho, phi), skiprows=1, usecols=1)
        X = np.loadtxt(data_dir + '/x_p{}_n{}_rho{}_phi{}.txt'.format(p, n, rho, phi), skiprows=1, usecols=np.arange(1, p+1))
    else:
        np.random.seed(seed)
        noise = np.random.normal(loc=0, scale=np.sqrt(phi), size=[n])  # noise, with std phi
        sigma = rho * np.ones((p, p))
        np.fill_diagonal(sigma, 1)
        X = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=n)  # X, with correlation rho
        theta = np.zeros(p)
        theta[-5:] = truetheta  # beta
        Y = np.matmul(X, theta) + noise
    return Y, X, theta


class HardConcreteSampler(nn.Module):
    """
    Sampler for Hard concrete random variable used for L0 gate
    """
    def __init__(self, p, scale, temparature, init=np.log(0.1/0.9)):
        super(HardConcreteSampler, self).__init__()
        self.p = p
        self.zeta, self.gamma, self.beta = scale, -(scale - 1), temparature # 1.1, -0.1, 9/10  #2/3
        self.gamma_zeta_logratio = np.log(-self.gamma / self.zeta)
        self.logalpha = nn.Parameter(init * torch.ones(p)) # np.log(0.1/0.9)
        # self.logalpha.data.uniform_(np.log(0.2), np.log(10))

    def forward(self, repeat):
        qz = torch.sigmoid(self.logalpha - self.beta * self.gamma_zeta_logratio)   # qz = p(z>0)
        u = Uniform(0, 1).sample([repeat, self.p]).cuda()
        s = torch.sigmoid((torch.log(u / (1 - u)) + self.logalpha) / self.beta)
        if self.training:
            z = torch.clamp((self.zeta - self.gamma) * s + self.gamma, 0, 1)
        else:
            z = torch.clamp(torch.sigmoid(self.logalpha) * (self.zeta - self.gamma) + self.gamma, 0, 1).expand_as(u)
        return z, qz

class LearnableHardConcreteSampler(nn.Module):
    """
    Sampler for Hard concrete random variable used for L0 gate
    """
    def __init__(self, p, temparature, init=np.log(0.1/0.9)):
        super(LearnableHardConcreteSampler, self).__init__()
        self.p = p
        self.beta = temparature
        # self.temparature = nn.Parameter(5. * torch.ones(p))
        # self.zeta, self.gamma = scale, -(scale - 1)  # 1.1, -0.1, 9/10  #2/3
        # self.gamma_zeta_logratio = np.log(-self.gamma / self.zeta)

        self.scale = nn.Parameter(-0. * torch.ones(p))
        self.logalpha = nn.Parameter(init * torch.ones(p))  # np.log(0.1/0.9)

    def forward(self, repeat):
        scale = F.softplus(self.scale) + 1#.clamp(min=0.1, max=0.1)  # 1.1, -0.1, 9/10  #2/3
        # scale = torch.exp(self.scale) + 1
        self.zeta, self.gamma = scale, -(scale - 1)
        self.gamma_zeta_logratio = torch.log(-self.gamma / self.zeta)

        qz = torch.sigmoid(self.logalpha - self.beta * self.gamma_zeta_logratio)   # qz = p(z>0)
        u = Uniform(0, 1).sample([repeat, self.p]).cuda()
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

        return self.forward((spl.cuda(), lgd.cuda(), context))

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

        return h, logdet, x


class GaussianAlternative(nn.Module):
    """
    Independent Gaussian alternative distribution
    """
    def __init__(self, p):
        super(GaussianAlternative, self).__init__()
        self.p = p
        self.mean = nn.Parameter(torch.rand(p)).cuda()
        self.logvar = nn.Parameter(torch.rand(p)).cuda()

    def forward(self, inputs):
        return inputs * (0.5 * self.logvar).exp() + self.mean

    def sample(self, n):
        noise = torch.rand(n, self.p).cuda()
        x = self.forward(noise)
        qlogq = - 0.5*self.logvar - (x - self.mean) ** 2 / (2 * self.logvar.exp())
        return x, -qlogq, noise


class SpikeAndSlabSampler(nn.Module):
    """
    Spike and slab sampler with Dirac spike and FlowAlternative slab.
    """
    def __init__(self, p, alternative_sampler, scale, temparature, init):
        super(SpikeAndSlabSampler, self).__init__()
        self.p = p
        self.alternative_sampler = alternative_sampler(p)
        if scale == 'learned':
            self.z_sampler = LearnableHardConcreteSampler(p, temparature, init)
        else:
            self.z_sampler = HardConcreteSampler(p, scale, temparature, init)

    def forward(self, repeat):
        theta, logdet, gaussians = self.alternative_sampler.sample(n=repeat)
        z, qz = self.z_sampler(repeat)
        out = z * theta
        logq = normal_log_pdf(gaussians, torch.zeros(self.p).cuda(), torch.zeros(self.p).cuda())
        return out, theta, logq, logdet, z, qz


class LinearModel(nn.Module):
    """
    Wrap around SpikeAndSlabSampler for a linear regression model.
    compare with: https://www.tandfonline.com/doi/pdf/10.1080/01621459.2015.1130634?needAccess=true
    """
    def __init__(self, p, scale, temparature, init, bias=False, alternative_sampler=FlowAlternative):  # FlowAlternative or GaussianAlternative
        super(LinearModel, self).__init__()
        self.add_bias = bias
        if self.add_bias:
            self.bias = nn.Parameter(torch.ones(p[1]))
            self.bias.data.uniform_(-0.1, 0.1)

        if isinstance(p, tuple):
            p = p[0] * p[1]
        self.alternative_sampler = alternative_sampler
        self.sampler = SpikeAndSlabSampler(p, alternative_sampler, scale, temparature, init)

    def forward(self, x):
        self.Theta, self.theta, self.logq, self.logdet, self.z, self.qz = self.sampler(repeat=16)
        if self.add_bias:
            out = x.matmul(self.Theta.mean(dim=0)) + self.bias
        else:
            out = x.matmul(self.Theta.permute(1, 0))  # x.matmul(self.Theta.mean(dim=0))  # (100, repeat)
        return out

    def kl(self, alter_prior, tau):
        p = 0.1
        qz = self.qz.expand_as(self.Theta)  # .clamp(min=1e-3, max=0.999)
        kl_z = qz * torch.log(qz.clamp(min=1e-10) / p) + (1 - qz) * torch.log((1 - qz).clamp(1e-10) / (1 - p))
        qlogq = self.logq - self.logdet

        if alter_prior == 'mom':
            qlogp = mom_log_pdf(self.theta, tau).clamp(min=np.log(1e-10))  # tau = 0.358
        elif alter_prior == 'imom':
            qlogp = imom_log_pdf(self.theta, tau).clamp(min=np.log(1e-10)) # 0.133
        elif alter_prior == 'pemom':
            qlogp = pemom_log_pdf(self.theta, tau).clamp(min=np.log(1e-10))
        elif alter_prior == 'Gaussian':
            qlogp = normal_log_prob(self.theta, var=10.)  #.clamp(min=np.log(1e-10))
        kl_beta = qlogq - qlogp
        kl = (kl_z + qz*kl_beta).sum(dim=1).mean()
        return kl, qlogp.mean(), qlogq.mean()



def loglike(y_hat, y):
    ll = - (y_hat.permute(1,0)-y)**2/(2*1**2)
    return ll.sum(dim=1).mean()



def train(Y, X, truetheta, epoch, alter_prior, tau, rep, lr, scale, temparature, init, p, phi, result_dir):
    linear = LinearModel(p, scale, temparature, init).cuda()
    # optimizer = optim.SGD(linear.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(linear.parameters(), lr=lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000], gamma=0.5,
                                                     last_epoch=-1)  # 10, 20, 30
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9, last_epoch=-1)

    sse_list = []
    sse_theta_list = []
    sse_list_train = []
    sse_nonzero_list = []
    sse_zero_list = []
    for i in range(epoch):
        linear.train()
        optimizer.zero_grad()

        # forward pass and compute loss
        y_hat = linear(X)
        nll = -loglike(y_hat, Y)
        kl, qlogp, qlogq = linear.kl(alter_prior, tau)
        loss = nll + (0.05*X.shape[1]) * kl / X.shape[0]

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        scheduler.step()

        sse_train = ((y_hat.permute(1,0) - Y) ** 2).mean().cpu().detach().numpy()
        sse_list_train.append(sse_train)


        with torch.no_grad():
            linear.eval()
            y_hat = linear(X)
            y_hat = y_hat.mean(dim=1)
            # get sse
            sse = ((Y-y_hat) ** 2).mean().cpu().detach().numpy()
            sse_list.append(sse)

            p = X.shape[1]
            square_error = ((linear.Theta.mean(dim=0).cpu() - torch.tensor(truetheta, dtype=torch.float)) ** 2)
            sse_theta = square_error.sum().item()

            sse_nonzero, sse_zero = square_error[(p-5):].sum().item(), square_error[:(p-5)].sum().item()
            sse_nonzero_list.append(sse_nonzero)
            sse_zero_list.append(sse_zero)

            sse_theta_list.append(sse_theta)
            if epoch > 100:  # sse <= np.asarray(sse_list).min() and
                sse_zero_best = sse_zero
                sse_nonzero_best = sse_nonzero
                min_sse_theta = sse_theta
                min_sse = sse
                best_theta = linear.Theta.mean(dim=0).detach().cpu().numpy()

        # print intermediet results
        if i % 200 == 0 or i == epoch-1:
            z = linear.z.mean(dim=0).cpu().numpy()
            # qz = linear.qz
            print('\n', 'repeat', rep, 'epoch', i, 'last 5 responses:', y_hat[-5:].round().tolist(), Y[-5:].round().tolist())
            print('sse_theta:{:.3f}, min_sse_theta:{:.3f}'.format(sse_theta, min_sse_theta))
            print('sse_nonzero:{:.3f}, best: {}'.format(sse_nonzero, sse_nonzero_best))
            print('sse_zero:{:.3f}, best:{}'.format(sse_zero, sse_zero_best))
            print('est.thetas: {}, \n est z:{}'.format(linear.Theta.mean(dim=0)[-10:].cpu().numpy().round(3), z[-50:].round(3)))
            print('epoch {}, z min: {}, z mean: {}, non-zero: {}'.format(i, z.min(), z.mean(), z.nonzero()[0].shape))
            print('nll, kl, qlogp', nll.tolist(), kl.tolist(), qlogp.item())
            threshold = utils.search_threshold(z, 0.05)
            print('threshold', threshold)
            print('number of cov above threshold', np.sum(z>threshold))
            print('sse:{}, min_sse:{}, \n'.format(sse, min_sse))


    theta, _, _, _, _, _ = linear.sampler(repeat=2000)
    theta = theta.detach().cpu().numpy()
    # with open(result_dir + '/{}_theta_posterior_p{}_phi{}_repeat{}.pkl'.format(alter_prior, p, phi, rep), 'wb') as f:
    #     pkl.dump(theta, f)
    # #
    # with open(result_dir + '/{}_sse_zero_list_p{}_phi{}_repeat{}.pkl'.format(alter_prior, p, phi, rep), 'wb') as f:
    #     pkl.dump(sse_zero_list, f)
    # with open(result_dir + '/{}_sse_nonzero_list_p{}_phi{}_repeat{}.pkl'.format(alter_prior, p, phi, rep), 'wb') as f:
    #     pkl.dump(sse_nonzero_list, f)

    return linear, best_theta, sse_nonzero_best, sse_zero_best, min_sse_theta


def test(Y, X, model):
    model.eval()
    y_hat = model(X)
    y_hat = y_hat.mean(dim=1)
    sse = ((y_hat - Y) ** 2).cpu().mean().detach().numpy()

    print('test SSE:{}'.format(sse))
    if model.z.nonzero().shape[0] < 10:
        print(model.z.nonzero(), model.z[-5:].tolist())


config = {
    'save_model': False,
    'save_model_dir': '/extra/yadongl10/git_project/nlpresult'
}


def main(config):
    n = 100
    rep = 30
    alter_prior = 'imom'  # Gaussian, imom

    result_dir = '/extra/yadongl10/git_project/nlpresult/0205/adam005_init0_tau10_rho025_notlearned_scale1.1_q_gauss/{}'.format(alter_prior)  # gau_alter_

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    epochs = 10000

    tau = 10
    rho = 0.25
    seed = 100 + np.arange(rep)
    lr = 0.01   # sgd 0.0001  adam 0.005
    scale = 'learned'  # 1.1
    temparature = 9/10
    init = 0.  #np.log(0.1/0.9)

    for phi in [1]:  # [1, 4, 8]
        for p in [500, 1000, 1500]:  # [100, 500, 1000]  500, 1000, 1500
            sse_theta_ls = []
            for i in range(rep+2):
                print('CONFIG: n {}, p {}, phi {}, alter_prior {}, seed {}, lr {}, temparature {}'.format(n, p, phi, alter_prior, seed[i], lr, temparature))
                Y, X, truetheta = generate_data(n, p, phi, rho=rho, seed=seed[i], load_data=False)
                Y, X = torch.tensor(Y, dtype=torch.float).cuda(), torch.tensor(X, dtype=torch.float).cuda()
                linear, best_theta, sse_nonzero_best, sse_zero_best, sse_theta = train(Y, X, truetheta, epoch=epochs,
                                                                                    alter_prior=alter_prior, tau=tau, rep=i, lr=lr, p=p,
                                                                                    scale=scale, temparature=temparature, init=init, phi=phi, result_dir=result_dir)

                sse_theta_ls.append([sse_theta, sse_nonzero_best, sse_zero_best])
                test(Y, X, linear)

            #     with open(result_dir + '/p{}_phi{}_repeat{}.pkl'.format(p, phi, i), 'wb') as f:
            #         pkl.dump(best_theta, f)
            #
            # with open(result_dir + '/p{}_phi{}_sse_theta_ls_tau{}.pkl'.format(p, phi, tau), 'wb') as f:
            #     pkl.dump(sse_theta_ls, f)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Flow spike and slab')
    # parser.add_argument('--p',type=int, default=100, help='number of covariates (default: 100)')
    parser.add_argument('--result_dir', type=str,
                        default='/extra/yadongl10/git_project/nlpresult/0203')
    main(config)


