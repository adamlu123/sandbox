import argparse
import torch.utils.data
from torch import nn, optim
from torch.distributions import Uniform, Normal
from torchvision import datasets, transforms
import pickle as pkl
import torch
import torch.nn.functional as F
import numpy as np
from nonlocal_vi.LinearRegressionFlow import FlowAlternative
from ContLearn_VAE import HardConcreteSampler


def normal_log_pdf(x, mu, logvar):
    return -0.5 * logvar - 0.5 * np.log(2*np.pi) - (x - mu)**2/(2*logvar.exp())


def mom_log_pdf(x, phi=1, tau=0.358):  # 0.358
    return ((x**2).clamp(min=1e-10)).log() - np.log(tau) - np.log(np.sqrt(2*np.pi*tau*phi)) - x**2/(2*tau*phi)


def imom_log_pdf(x, phi=1, tau=3):  #0.133
    # x = x.sign() * x.abs().clamp(min=1e-5, max=1e5)
    return 0.5*np.log(tau*phi/np.pi) - (x**2).log() - (tau*phi) / (x**2) #.clamp(min=1e-10, max=1e10)

def pemom_log_pdf(x, phi=1, tau=0.358):
    return (np.sqrt(2) - (tau*phi) / x**2) - np.log(np.sqrt(2*np.pi*tau*phi)) - x**2/(2*tau*phi)


def normal_log_prob(x, var=10):
    return np.log(1/np.sqrt(2*np.pi)) - 0.5 * np.log(var) - x**2/(2*var)


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = 'cuda' # torch.device("cuda" if args.cuda else "cpu")


def filter_trials(trial_info):
    """
    Get indices of correct in-sequence trials of odors A to D.
    """
    rat_correct = trial_info[:, 0] == 1
    in_sequence = trial_info[:, 1] == 1
    not_odor_e = trial_info[:, 3] < 5
    select = rat_correct & not_odor_e
    return select


def clean_data(trial_info, spike_data, lfp_data, training=False):
    """
    Clean up trials, remove cells that do not fire, and label targets.
    """
    trial_indices = filter_trials(trial_info)
    spike_data = spike_data[trial_indices]
    lfp_data = lfp_data[trial_indices]
    total_spike = np.sum(np.sum(spike_data[:, :, 100:300], axis=2), axis=0)
    spike_data = spike_data[:, total_spike > 0, :]
    target = trial_info[trial_indices, 1]  # whether inseq
    odor = trial_info[trial_indices, 3] - 1

    if training:
        spike_data = np.mean(spike_data[:, :, 200:300], axis=2)  # 210:235
    return target, spike_data, lfp_data, odor


def prepare_odor_data(subset = 'SuperChris'):
    data_dir = '/extra/yadongl10/data/rat_odor/Processed/'
    st = np.load(data_dir + subset + '/{}_spike_data_binned.npy'.format(subset.lower()))
    info = np.load(data_dir + subset +'/{}_trial_info.npy'.format(subset.lower()))
    lfp_data = np.load(data_dir + subset +'/{}_lfp_data_sampled.npy'.format(subset.lower()))
    target, spike_data, lfp_data, odor = clean_data(info, st, lfp_data, training=True)
    return target, spike_data, lfp_data, odor

target, spike_data, lfp_data, odor = prepare_odor_data(subset = 'SuperChris')
print(spike_data.shape, lfp_data.shape, target.shape)



def train(epoch, model, optimizer, data):

    model.train()
    data = data.to(device)
    optimizer.zero_grad()

    # forward pass
    recon_batch = model(data).mean(dim=0)   # (16, 168, input_dim) -> (168, input_dim)

    # compute elbo loss and print intermediate results
    nll = (recon_batch - data.view(-1, 42))**2     # (128, input_dim)
    nll = nll.sum(1)    # (128)
    kl, qlogp, qlogq = model.kl(phi=1)
    loss = nll + kl
    loss = loss.mean()
    loss.backward()
    optimizer.step()
    print('Train Epoch: {}, Loss: {:.6f}'.format(epoch, loss.item()))
    return loss.item()


class FeatureSelectNet(nn.Module):
    def __init__(self, input_dim, add_bias=False, alternative_sampler=FlowAlternative):
        super(FeatureSelectNet, self).__init__()
        self.input_dim = input_dim
        self.add_bias = add_bias
        if add_bias:
            self.bias = nn.Parameter(torch.ones(input_dim))
            self.bias.data.uniform_(-0.1, 0.1)

        self.alternative_sampler = alternative_sampler(p=input_dim)
        self.z_sampler = HardConcreteSampler(input_dim)
        self.attention = nn.Linear(5, input_dim)

    def forward(self, x, group):
        logalpha = self.attention(group)
        self.z, self.qz = self.z_sampler(repeat=16, logalpha=logalpha)  # z.shape = (repeat, batch, p)  qz.shape=(p)
        self.theta, self.logdet, gaussians = self.alternative_sampler.sample(n=16)
        self.logq = normal_log_pdf(gaussians, torch.zeros(self.input_dim).cuda(), torch.zeros(self.input_dim).cuda())
        self.Theta = self.theta * self.z.permute(1,0,2)
        self.Theta = self.Theta.permute(1,0,2)

        if self.add_bias:
            out = (x * self.Theta + self.bias).mean(dim=0)
        else:
            out = (x * self.Theta).mean(dim=0)
        return torch.sigmoid(out.sum(dim=1)) # (168, 42) -> (168)

    def kl(self, phi=1, alter_prior='mom', p=0.5):
        qz = self.qz.expand_as(self.Theta)
        kl_z = qz * torch.log(qz.clamp(min=1e-10) / p) + (1 - qz) * torch.log((1 - qz).clamp(1e-10) / (1 - p))
        qlogq = self.logq - self.logdet

        if alter_prior == 'mom':
            qlogp = mom_log_pdf(self.theta, phi, tau=1).clamp(min=np.log(1e-10))  # tau = 0.358
        elif alter_prior == 'imom':
            qlogp = imom_log_pdf(self.theta, tau=1).clamp(min=np.log(1e-10))  # 0.133
        elif alter_prior == 'pemom':
            qlogp = pemom_log_pdf(self.theta)  #.clamp(min=np.log(1e-10))
        elif alter_prior == 'Gaussian':
            qlogp = normal_log_prob(self.theta, var=1.)  #.clamp(min=np.log(1e-10))
        kl_beta = qlogq - qlogp
        kl = (kl_z + qz*kl_beta.unsqueeze(1).repeat(1, self.Theta.shape[1], 1)).sum(dim=2).mean()  # (16, 189, 42) ->
        return kl, qlogp.mean(), qlogq.mean()


def train(epoch, model, optimizer, data, target, odor, config):

    model.train()
    data = data.to(device)
    optimizer.zero_grad()

    # forward pass
    pred = model(data, odor)
    pred_class = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
    acc = (pred_class==target).sum().float() / target.shape[0]

    # compute elbo loss and print intermediate results
    # criterion = torch.nn.BCEWithLogitsLoss()
    # nll = criterion(pred, target)     # (128, input_dim)
    nll = - (target * pred.clamp(min=1e-10).log() + (1 - target) * (1 - pred).clamp(min=1e-10).log())    # (128)
    kl, qlogp, qlogq = model.kl(1, config['alter_prior'], config['p'])
    loss = nll.sum() + kl
    loss = loss.mean()
    loss.backward()
    optimizer.step()
    print('Train Epoch: {}, Loss: {:.3f}, kl:{:3f}, acc:{:3f}'.format(epoch, loss.item(), kl.item(), acc.item()))
    return loss.item(), model


def test(model, data, test_target, odor, odor_onehot, config):
    """

    :return: the learned mask by odor:
    """
    model.eval()
    pred = model(data, odor_onehot)
    z = model.z.mean(dim=0).detach().cpu().numpy()

    result = np.zeros((4, z.shape[1]))
    for i in range(4):
        result[i, :] = z[odor==i,:].mean(axis=0)  # (latent_dim)

    pred_class = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
    acc = (pred_class == test_target).sum().float() / test_target.shape[0]
    nll = - (test_target * pred.clamp(min=1e-10).log() + (1 - test_target) * (1 - pred).clamp(min=1e-10).log())  # (128)
    kl, qlogp, qlogq = model.kl(1, config['alter_prior'], config['p'])
    loss = nll.sum() + kl

    print('test test loss: {:.3f}  acc :{:.3f}'.format(loss.item(), acc.item()))
    return result, loss.item(), acc.item()


def diff(full_set, second):
    second = set(second)
    return [item for item in full_set if item not in second]


def main(rep, config):
    ## prepare data
    target, spike_data, lfp_data, odor = prepare_odor_data(subset='SuperChris')  # ((168,), (168, 42), (168, 400, 21))
    n = target.shape[0]
    spike_data = torch.tensor(spike_data, dtype=torch.float).to(device) * 100
    test_size = n // 5
    test_id = np.arange(rep*test_size, (rep+1)*test_size)
    train_id = diff(np.arange(n), test_id)

    train_spike_data, test_spike_data = spike_data[train_id, :], spike_data[test_id, :]
    # target = torch.tensor((np.arange(2) == target[:, None]).astype(np.float32)).to(device)
    target = torch.tensor(target, dtype=torch.float).to(device)
    train_target, test_target = target[train_id], target[test_id]
    odor_onehot = torch.tensor((np.arange(5) == odor[:, None]).astype(np.float32), dtype=torch.float).to(device)
    train_odor_onehot, test_odor_onehot = odor_onehot[train_id, :], odor_onehot[test_id, :]

    # define model
    model = FeatureSelectNet(input_dim=spike_data.shape[1], add_bias=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    print(model.parameters)

    # start training
    train_loss_list = []
    for epoch in range(config['epochs'] + 1):
        loss, model = train(epoch, model, optimizer, train_spike_data, train_target, train_odor_onehot, config)
        if epoch % 250 == 0:
            mask_by_odor, test_loss, test_acc = test(model, test_spike_data, test_target, odor[test_id], test_odor_onehot, config)
            print(mask_by_odor.round(1))

    # save results
    result_dir = '/extra/yadongl10/git_project/nlpresult/rat_exp/cross_valid'
    # np.savetxt(result_dir + '/SuperChris_selection_by_odor_prior{}_no_bias.pkl'.format('mom'), )
    with open(result_dir + '/SuperChris_selection_by_odor_p{}_prior{}_repeat{}.pkl'.format(config['p'], config['alter_prior'], rep), 'wb') as f:
        pkl.dump(mask_by_odor, f)
    with open(result_dir + '/SuperChris_selection_by_odor_p{}_prior{}_testacc.txt'.format(config['p'], config['alter_prior']), 'a') as f:
        f.write('\n' + str(test_acc))
    with open(result_dir + '/SuperChris_selection_by_odor_p{}_prior{}_testloss.txt'.format(config['p'], config['alter_prior']), 'a') as f:
        f.write('\n' + str(test_loss))


if __name__ == "__main__":
    # 5-fold CV
    repeat = 5
    config = {'alter_prior': 'imom',  # Gaussian, mom, imom
              'p': 0.1,
              'epochs': 1000}
    for rep in range(repeat):
        print('start rep', rep, '\n')
        main(rep, config)