import argparse
import torch
import torch.nn.functional as F
import numpy as np
import utils
from utils import log
from ContLearn_VAE import NonLocalVAE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
parser.add_argument('--epochs', type=int, default=5000, metavar='N',
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
    select = rat_correct & in_sequence & not_odor_e
    return select


def clean_data(trial_info, spike_data, lfp_data, training=False):
    """
    Clean up trials, remove cells that do not fire, and label targets.
    """
    trial_indices = filter_trials(trial_info)
    spike_data = spike_data[trial_indices]
    lfp_data = lfp_data[trial_indices]
    total_spike = np.sum(np.sum(spike_data[:, :, 200:300], axis=2), axis=0)
    spike_data = spike_data[:, total_spike > 0, :]
    target = trial_info[trial_indices, 3] - 1
    if training:
        spike_data = np.mean(spike_data[:, :, 200:300], axis=2)  # 210:235
    return target, spike_data, lfp_data


def prepare_odor_data(subset = 'SuperChris'):
    data_dir = '/extra/yadongl10/data/rat_odor/Processed/'
    st = np.load(data_dir + subset + '/{}_spike_data_binned.npy'.format(subset.lower()))
    info = np.load(data_dir + subset +'/{}_trial_info.npy'.format(subset.lower()))
    lfp_data = np.load(data_dir + subset +'/{}_lfp_data_sampled.npy'.format(subset.lower()))
    target, spike_data, lfp_data = clean_data(info, st, lfp_data, training=True)
    return target, spike_data, lfp_data

target, spike_data, lfp_data = prepare_odor_data(subset = 'SuperChris')
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


def get_latent(data):
    # model.eval()
    mu, logvar, logalpha = model.encoder(data.float())
    latent, _, _, _, z, qz = model.latent_sampler(1, mu, logvar, logalpha)
    return latent, z, qz


def test(model, spike_data, target):
    """

    :return: the learned probability of latent units being activated by odor: [5, 10]
    """
    latent, z, qz = get_latent(spike_data)  # qz = (168, latent_dim)
    latent, z, qz = latent.mean(dim=0).detach().cpu().numpy(), \
                    z.mean(dim=0).detach().cpu().numpy(), \
                    qz.detach().cpu().numpy()

    result = np.zeros((4, qz.shape[1]))
    for i in range(4):
        result[i, :] = qz[target==i,:].mean(axis=0)  # (latent_dim)
    return result, qz, latent



def plot(result, qz, latent, target, epoch):
    c = ['r', 'blue', 'g', 'y']
    label = ['A', 'B', 'C', 'D']
    qz_reduced = TSNE(n_components=2, random_state=0).fit_transform(qz)
    for i in range(4):
        plt.scatter(qz_reduced[target == i, 0], qz_reduced[target == i, 1], c=c[i], label=label[i])
    plt.title('Clustering of latent space activity by TSNE plot')
    plt.xlabel('dimension 1')
    plt.ylabel('dimension 2')

    # for i in range(4):
    #     plt.scatter(qz[target==i, 0], latent[target==i, 1], c=c[i], label=label[i])
    plt.legend()
    plt.savefig('qz_ep{}.png'.format(epoch))
    plt.close()

if __name__ == "__main__":
    target, spike_data, lfp_data = prepare_odor_data(subset='SuperChris')  # ((168,), (168, 42), (168, 400, 21))
    spike_data = torch.tensor(spike_data, dtype=torch.float).to(device) * 100

    model = NonLocalVAE(input_dim=spike_data.shape[1], latent_dim=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    print(model.parameters)

    train_loss_list = []
    for epoch in range(args.epochs + 1):
        # train_loss_list += train(epoch, model, optimizer, spike_data)
        train(epoch, model, optimizer, spike_data)
        if epoch % 500 == 0:
            result, qz, latent = test(model, spike_data, target)
            print(result)
            plot(result, qz, latent, target, epoch)


        with open('latent.pkl', 'wb') as f:
            pkl.dump(latent, f)
        with open('qz.pkl', 'wb') as f:
            pkl.dump(qz, f)

    # with open('vae_bayes_loss.pkl'.format(epoch), 'wb') as f:
    #     pkl.dump(train_loss_list, f)