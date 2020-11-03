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
import h5py

from __future__ import print_function
import argparse
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pickle as pkl


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
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



def get_train_loader(subset=0, batch_size=128):
    """
    :param subset:
    :param batch_size:
    :return: train_loader
    """
    with h5py.File('MNSIT_by_class.h5', 'r') as f:
        data = np.asarray(f[str(subset)+'_data'])
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_loader


# TODO: 1. determine args.epoch for each class. 2. determine how many tiems replay. 3. k NN classifer
def test(encoder, k, num_each_class):
    encoder.eval()  # TODO: notice there is a difference in L0 gate
    data = get_test_data(num_each_class=num_each_class)
    latent = encoder(data)

    knn = kNNClassifer(data)
    pred = knn(latent, k=k)
    label = torch.arange(10).unsqueeze(1).repeat(1, num_each_class).view(-1)
    acc = get_accuracy(pred, label).tolist()
    print('test acc of {}-NN is:'.format(k, acc))


def get_test_data(num_each_class = 1000):
    test_data = np.zeros(10*num_each_class, 784)
    for j in range(10):
        with h5py.File('MNSIT_by_class.h5', 'r') as f:
            test_data[j*num_each_class, :] = np.asarray(f[str(j)+'_data'][:num_each_class]).reshape(-1, 784)
    return test_data


class kNNClassifer(nn.Module):
    def __int__(self, data):
        super(kNNClassifer, self).__init__()
        self.data = data
        self.num_each_class = self.data.shape[0]/10

    def forward(self, inputs, k):
        inputs = inputs.unsqueeze(1).repeat(1, self.data.shape[0], 1)
        distance = ((inputs - self.data) ** 2).sum(dim=2)  # shape=(num_sample, num_base_sample)
        _, index = torch.topk(distance, k, dim=1)  # index shape=(num_sample, k)
        index = index // self.num_each_class
        pred, _ = torch.mode(index, dim=1)
        return pred


def get_accuracy(pred, label):
    """
    :param pred: 1-D tensor
    :param label: 1-D tensor
    :return:
    """
    return (pred==label).sum() / label.shape[0]


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

    def forward(self, x):
        x = x.view(-1, 784)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std






# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def loss_function_encoder(update_z, mu, logvar):
    nll = - 0.5 * logvar - (update_z - mu) ** 2 / (2 * logvar.exp())
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return nll.sum() + KLD, KLD


def train(epoch, train_loader):
    model_enc.train()
    model_dec.train()

    train_loss = 0
    train_loss_list = []
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optim_enc.zero_grad()
        optim_dec.zero_grad()

        # forward pass
        mu, logvar = model_enc(data)
        z = reparameterize(mu, logvar)
        z = z.data.requires_grad_()
        recon_batch = model_dec(z)

        # update decoder and get update_z
        BCE = F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction='sum')
        BCE.backward()
        update_z = z + 1e-3 * z.grad
        optim_dec.step()
        z.grad.zero_()

        # update encoder
        loss_enc, KLD = loss_function_encoder(update_z, mu, logvar)
        loss_enc.backward()
        optim_enc.step()

        # compute elbo loss and print intermediate results
        loss = BCE + KLD
        train_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
        train_loss_list.append(loss.item())

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss_list


if __name__ == "__main__":
    model_enc = Encoder().to(device)
    model_dec = Decoder().to(device)

    # model = VAE().to(device)
    optim_enc = optim.Adam(model_enc.parameters(), lr=1e-3)
    optim_dec = optim.Adam(model_dec.parameters(), lr=1e-3)

    train_loss_list = []
    for i in range(10):
        train_loader = get_train_loader(subset=i, batch_size=128)
        for epoch in range(1, args.epochs + 1):
            train_loss_list += train(epoch, train_loader)

        test(model_enc, k=10, num_each_class=1000)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                'results/sample_' + str(epoch) + '.png')
    with open('vae_bayes_loss.pkl'.format(epoch), 'wb') as f:
        pkl.dump(train_loss_list, f)

