import argparse
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch.utils.data
from torch import nn, optim
from torch.distributions import Uniform, Normal
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pickle as pkl
from nonlocal_vi.LinearRegressionFlow import FlowAlternative
from ContLearn_VAE import HardConcreteSampler


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
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


def filter_trials(trial_info):
    rat_correct = trial_info[:, 0] == 1
    in_sequence = trial_info[:, 1] == 1
    not_odor_e = trial_info[:, 3] < 5
    select = rat_correct & in_sequence & not_odor_e
    return select


def prepare_data(rat_name = 'SuperChris'):
    # load data
    data_dir = '/extra/yadongl10/data/rat_odor/Processed/'
    spike_data_binned = np.load(data_dir + rat_name + '/{}_spike_data_binned.npy'.format(rat_name.lower()))
    lfp_data_sampled = np.load(data_dir + rat_name + '/{}_lfp_data_sampled.npy'.format(rat_name.lower()))
    lfp_data_sampled = np.swapaxes(lfp_data_sampled, 1, 2)
    trial_info = np.load(data_dir + rat_name + '/{}_trial_info.npy'.format(rat_name.lower()))
    # process data
    trial_indices = filter_trials(trial_info)
    decoding_start = 210  # 210
    decoding_end = decoding_start + 25  # 25
    # scale data first
    spike_data_binned = spike_data_binned[trial_indices, :, :]
    spike_data_binned = (spike_data_binned - np.mean(spike_data_binned)) / np.std(spike_data_binned)
    lfp_data_sampled = lfp_data_sampled[trial_indices, :, :]
    decoding_data_spike = spike_data_binned[:, :, decoding_start:decoding_end]
    decoding_data_lfp = lfp_data_sampled[:, :, decoding_start:decoding_end]
    odor = (trial_info[trial_indices, 3] - 1).astype(int) # np_utils.to_categorical()  # odor info
    # organize tetrode data
    if rat_name.lower() == 'superchris':
        tetrode_ids = [1, 10, 12, 13, 14, 15, 16, 18, 19, 2, 20, 21, 22, 23, 3, 4, 5, 6, 7, 8, 9]
        tetrode_units = {1:3, 10:0, 12:1, 13:8, 14:4, 15:6, 16:1, 18:0, 19:4, 2:3,
                         20:0, 21:1, 22:5, 23:7, 3:0, 4:0, 5:0, 6:0, 7:1, 8:1, 9:1}
    elif rat_name.lower() == 'stella':
        tetrode_ids = [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9]
        tetrode_units = {10:1, 12:0, 13:5, 14:7, 15:4, 16:8, 17:0, 18:0, 19:1, 20:0,
                     21:1, 22:1, 23:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:13, 8:4, 9:4}
    elif rat_name.lower() == 'buchanan':
        tetrode_ids = [10, 12, 13, 15, 16, 17, 18, 19, 1, 20, 21, 22, 23, 2, 4, 5, 6, 7, 8, 9]
        tetrode_units = {10:0, 12:0, 13:9, 15:6, 16:0, 17:4, 18:12, 19:15, 1:0,
                     20:0, 21:1, 22:13, 23:8, 2:2, 4:0, 5:6, 6:3, 7:0, 8:0, 9:0}
    elif rat_name.lower() == 'barat':
        tetrode_ids = [10, 12, 13, 14, 15, 16, 17, 18, 19, 1, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9]
        tetrode_units = {10:1, 12:20, 13:12, 14:7, 15:0, 16:0, 17:11, 18:0, 19:0, 1:1, 20:0, 21:9, 22:0,
                         23:1, 2:0, 3:11, 4:0, 5:4, 6:0, 7:1, 8:0, 9:14}
    elif rat_name.lower() == 'mitt':
        tetrode_ids = [12, 13, 14, 15, 16, 17, 18, 19, 1, 20, 21, 22, 23, 2, 3, 4, 5, 6, 7, 8, 9]
        tetrode_units = {12:16, 13:15, 14:4, 15:2, 16:6, 17:2, 18:12, 19:15, 1:0, 20:12, 21:0, 22:1,
                         23:4, 2:0, 3:4, 4:0, 5:0, 6:0, 7:3, 8:1, 9:7}
    tetrode_data = organize_tetrode(decoding_data_spike, decoding_data_lfp, tetrode_ids, tetrode_units)
    return tetrode_data, odor, tetrode_ids, tetrode_units, spike_data_binned, decoding_data_spike.mean(axis=2), lfp_data_sampled


def organize_tetrode(spike_data, lfp_data, tetrode_ids, tetrode_units, verbose=True):
    """
    Organize spike and LFP data by tetrode.
    :param spike_data: (3d numpy array) spike train data of format [trial, neuron, time]
    :param lfp_data: (3d numpy array ) LFP data of format [trial, tetrode, time]
    :param tetrode_ids: (list) of tetrode ids in the order of LFP data
    :param tetrode_units: (dict) number of neuron units on each tetrode
    :param verbose: (bool) whether to print each tetrode data shape
    :return: (list of 4d numpy arrays) each of format [trial, 1, neuron + tetrode, time]
    """
    all_tetrode_data = []
    i = 0

    for j, t in enumerate(tetrode_ids):
        k = tetrode_units[t]
        if k == 0:
            continue

        tetrode_lfp = np.expand_dims(lfp_data[:, j, :], axis=1)
        tetrode_spike = spike_data[:, i:(i + k), :]
        if len(tetrode_spike.shape) == 2:
            tetrode_spike = np.expand_dims(tetrode_spike, axis=1)

        tetrode_data = np.concatenate([tetrode_lfp, tetrode_spike], axis=1)
        tetrode_data = np.expand_dims(tetrode_data, axis=-1)

        all_tetrode_data.append(tetrode_data)

        if verbose:
            print('Current tetrode {t} with {k} neurons/units'.format(t=t, k=k))
            print(tetrode_data.shape)

        i += k
    return all_tetrode_data


# tetrode_data, odor, tetrode_ids, tetrode_units, spike_data_binned, lfp_data_sampled = prepare_data(subset='SuperChris')
# print(spike_data.shape, lfp_data.shape, target.shape)

# class tetrode_wise_logalpha(nn.Module):
#     def __init__(self, tetrode_group):
#         super(tetrode_wise_logalpha, self).__init__()
#         # [3, 1, 8, 4, 6, 1, 4, 3, 1, 5, 7, 1, 1, 1]
#         init = 0.
#
#         self.a1 = nn.Parameter(init * torch.ones(1))
#         self.a2 = nn.Parameter(init * torch.ones(1))
#         self.a3 = nn.Parameter(init * torch.ones(1))
#         self.a4 = nn.Parameter(init * torch.ones(1))
#         self.a5 = nn.Parameter(init * torch.ones(1))
#         self.a6 = nn.Parameter(init * torch.ones(1))
#         self.a7 = nn.Parameter(init * torch.ones(1))
#         self.a8 = nn.Parameter(init * torch.ones(1))
#         self.a9 = nn.Parameter(init * torch.ones(1))
#         self.a10 = nn.Parameter(init * torch.ones(1))
#         self.a11 = nn.Parameter(init * torch.ones(1))
#         self.a12 = nn.Parameter(init * torch.ones(1))
#         self.a13 = nn.Parameter(init * torch.ones(1))
#         self.a14 = nn.Parameter(init * torch.ones(1))
#         # self.logalpha_ls = []
#         # for p in tetrode_group:
#         #     self.logalpha_ls.append(nn.Parameter(0. * torch.ones(p)))
#
#     def forward(self):
#         group_ls = [self.a1, self.a2, self.a3, self.a4, self.a5, self.a6, self.a7, self.a8, self.a9, self.a10,
#                            self.a11, self.a12, self.a13, self.a14]
#         logalpha_ls = []
#         idx = [3, 1, 8, 4, 6, 1, 4, 3, 1, 5, 7, 1, 1, 1]
#         for i, a in enumerate(group_ls):
#             logalpha_ls.append(a.repeat(idx[i]))
#
#         return torch.cat(logalpha_ls, 0).to(device).unsqueeze(0)


def expand_z(z_group, qz_group, tetrode_group, odorwise=False):

    num_sample = z_group.shape[0]

    if not odorwise:
        z = torch.zeros(num_sample, 46).cuda()
        qz = torch.zeros(46).cuda()  # TODO is it correct?
        z_group, qz_group = z_group.squeeze(), qz_group.squeeze()
        ind = 0
        for i, a in enumerate(tetrode_group):
            z[:, ind:(ind+a)] = z_group[:, i].unsqueeze(1).repeat(1, a) # -> (16, 46)
            qz[ind:(ind + a)] = qz_group[i].repeat(a)
            ind += a
    else:
        z_group = z_group.permute(0, 2, 1).reshape(-1, 14, 4) # -> (16, 14*4, 1) -> (16, 14, 4)
        qz_group = qz_group.squeeze().reshape(-1, 4)  # -> (16, 4)
        z = torch.zeros(num_sample, 46, 4).cuda()
        qz = torch.zeros(46, 4).cuda()
        ind = 0
        for i, a in enumerate(tetrode_group):
            z[:, ind:(ind+a), :] = z_group[:, i, :].unsqueeze(1).repeat(1, a, 1)
            qz[ind:(ind+a), :] = qz_group[i, :].unsqueeze(0).repeat(a, 1).view(-1)

    return z, qz


class TetrodeSelectNet(nn.Module):
    def __init__(self, input_dim, output_dim, add_bias=False, alternative_sampler=FlowAlternative):
        super(TetrodeSelectNet, self).__init__()
        self.tetrode_group = [3, 1, 8, 4, 6, 1, 4, 3, 1, 5, 7, 1, 1, 1]
        num_tetrodes = len(self.tetrode_group)
        assert sum(self.tetrode_group) == input_dim
        self.input_dim = input_dim
        self.add_bias = add_bias

        self.output_dim = output_dim
        if add_bias:
            self.bias = nn.Parameter(torch.ones(output_dim))
            self.bias.data.uniform_(-0.1, 0.1)

        self.alternative_sampler = alternative_sampler(p=input_dim*output_dim)  # (46 * 4)
        self.z_sampler = HardConcreteSampler(num_tetrodes*output_dim)
        self.logalpha_group = nn.Parameter(-0.9 * torch.ones(num_tetrodes*output_dim))


    def forward(self, x):
        rep=32
        self.z_group, self.qz_goup = self.z_sampler(repeat=rep, logalpha=self.logalpha_group.view(1,-1))  # z.shape = (repeat, batch, p)  qz.shape=(p)
        self.z, self.qz = expand_z(self.z_group, self.qz_goup, self.tetrode_group)

        self.theta, self.logdet, gaussians = self.alternative_sampler.sample(n=rep)
        self.logq = normal_log_pdf(gaussians,
                                   torch.zeros(self.input_dim*self.output_dim).cuda(),
                                   torch.zeros(self.input_dim*self.output_dim).cuda())
        self.Theta = (self.theta.view(-1, 46, self.output_dim).squeeze() * self.z).permute(1,0)  # (16, 46, 4) -> (46 ,4)

        if self.add_bias:
            out = (x.matmul(self.Theta) + self.bias)
        else:
            out = x.matmul(self.Theta)
        return out.mean(dim=1, keepdim=True)  # (168, repeat) -> (168)

    def kl(self, phi=1, alter_prior='mom', p=0.01):
        qz = self.qz
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
        kl = (kl_z + qz*kl_beta).sum(dim=1).mean()  # (16, 189, 42) ->
        return kl, qlogp.mean(), qlogq.mean()



class OdorClassification(nn.Module):
    def __init__(self, input_dim, output_dim, SelectNet=TetrodeSelectNet):
        super(OdorClassification, self).__init__()
        self.SelectNet = SelectNet(input_dim, output_dim, add_bias=True)
        self.upper_layer1 = nn.Linear(1, 4)
        # self.upper_layer2 = nn.Linear(4, 4)
        # self.upper_layer3 = nn.Linear(4, 4)

    def forward(self, x):
        x = self.SelectNet(x)
        # x = F.relu(x)
        # x = torch.tanh(x)
        x = self.upper_layer1(x)
        # x = F.relu(self.upper_layer2(x))
        # x = F.relu(self.upper_layer3(x))
        return x

    def kl(self, phi, alter_prior, p):
        return self.SelectNet.kl(phi, alter_prior, p)


def train(epoch, model, optimizer, data, odor, config):

    model.train()
    data = data.to(device)
    odor = torch.tensor(odor).cuda()
    optimizer.zero_grad()

    # forward pass
    pred = model(data)
    pred_class = torch.argmax(F.softmax(pred, dim=1), dim=1)
    acc = (pred_class==odor).sum().float() / odor.shape[0]

    # compute elbo loss and print intermediate results
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    nll = criterion(pred, odor)     # (128, input_dim)
    kl, qlogp, qlogq = model.kl(1, config['alter_prior'], config['p'])
    loss = nll + kl
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Train Epoch: {}, Loss: {:.3f}, CE:{:.3f} kl:{:.3f}, acc:{:.3f}'.format(epoch, loss.item(), loss.item() - kl.item(), kl.item(), acc.item()))
    return loss.item(), model


def test(model, data, odor, odor_onehot, config):
    """
    :return: the learned mask by odor:
    """
    model.eval()
    data = data.to(device)
    odor = torch.tensor(odor).cuda()

    with torch.no_grad():
        pred = model(data)
        z = model.SelectNet.z_group.mean(dim=0).view(-1, config['output_dim']).detach().cpu().numpy()

        pred_class = torch.argmax(F.softmax(pred, dim=1), dim=1)
        acc = (pred_class == odor).sum().float() / odor.shape[0]
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        nll = criterion(pred, odor)
        kl, qlogp, qlogq = model.kl(1, config['alter_prior'], config['p'])
        loss = nll + kl

    print('test test loss: {:.3f}  acc :{:.3f}'.format(loss.item(), acc.item()))
    return z, loss.item(), acc.item()


def diff(full_set, second):
    second = set(second)
    return [item for item in full_set if item not in second]


def main(rep, config):
    ## prepare data
    tetrode_data, odor, tetrode_ids, tetrode_units, spike_data_binned, spike_data, lfp_data_sampled = prepare_data(rat_name='SuperChris')
    # odor, spike_data = odor[odor!=0]-1, spike_data[odor!=0]
    print(odor.shape, spike_data.shape)

    n = odor.shape[0]
    spike_data = torch.tensor(spike_data, dtype=torch.float).to(device) * 1

    test_size = n // 5
    test_id = np.arange(rep*test_size, (rep+1)*test_size)
    train_id = diff(np.arange(n), test_id)

    train_spike_data, test_spike_data = spike_data[train_id, :], spike_data[test_id, :]

    odor_onehot = torch.tensor((np.arange(4) == odor[:, None]).astype(np.float32), dtype=torch.float).to(device)
    train_odor_onehot, test_odor_onehot = odor_onehot[train_id, :], odor_onehot[test_id, :]

    # define model
    model = OdorClassification(input_dim=spike_data.shape[1], output_dim=config['output_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    print(model.parameters)

    # start training
    train_loss_list = []
    for epoch in range(config['epochs'] + 1):
        loss, model = train(epoch, model, optimizer, train_spike_data, odor[train_id], config)
        if epoch % 100 == 0:
            mask_by_odor, test_loss, test_acc = test(model, test_spike_data, odor[test_id], test_odor_onehot, config)
            print(mask_by_odor.round(2))

    # save results
    result_dir = '/extra/yadongl10/git_project/nlpresult/rat_exp/cross_valid'
    # np.savetxt(result_dir + '/SuperChris_selection_by_odor_prior{}_no_bias.pkl'.format('mom'), )
    # with open(result_dir + '/3odor_SuperChris_selection_by_tetrode_p{}_prior{}_repeat{}.pkl'.format(config['p'], config['alter_prior'], rep), 'wb') as f:
    #     pkl.dump(mask_by_odor, f)
    # with open(result_dir + '/3odor_SuperChris_selection_by_tetrode_p{}_prior{}_testacc.txt'.format(config['p'], config['alter_prior']), 'a') as f:
    #     f.write('\n' + str(test_acc))
    # with open(result_dir + '/3odor_SuperChris_selection_by_tetrode_p{}_prior{}_testloss.txt'.format(config['p'], config['alter_prior']), 'a') as f:
    #     f.write('\n' + str(test_loss))


if __name__ == "__main__":
    # 5-fold CV
    repeat = 5
    config = {'alter_prior': 'imom',  # Gaussian, mom, imom
              'p': 0.2,
              'output_dim': 1,
              'epochs': 10000}
    for rep in range(repeat):
        print('start rep', rep, '\n')
        main(rep, config)