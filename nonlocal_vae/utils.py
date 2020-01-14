import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform, Gamma, Bernoulli
import matplotlib.pyplot as plt
import numpy as np
import h5py


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


# TODO: 1. determine args.epoch for each class. 2. determine the replay schedule: time and amount. 3. k NN classifer
# def test(encoder, k, num_each_class):
#     encoder.eval()  # TODO: notice there is a difference in L0 gate
#     data = get_test_data(num_each_class=num_each_class)
#     latent = encoder(data)
#
#     knn = kNNClassifer(data)
#     pred = knn(latent, k=k)
#     label = torch.arange(10).unsqueeze(1).repeat(1, num_each_class).view(-1)
#     acc = get_accuracy(pred, label).tolist()
#     print('test acc of {}-NN is:'.format(k, acc))


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


def nlp_log_pdf(x, phi, tau=0.358):
    return (x**2).clamp(min=1e-2).log() - np.log(tau) - np.log(np.sqrt(2*np.pi*tau*phi)) - x**2/(2*tau*phi)


def normal_log_pdf(x, mu, logvar):
    return -0.5 * logvar - 0.5 * np.log(2*np.pi) - (x - mu)**2/(2*logvar.exp())


import torch
import torch.nn as nn

delta = 1e-6
softplus_ = nn.Softplus()
softplus = lambda x: softplus_(x) + delta
sigmoid_ = nn.Sigmoid()
sigmoid = lambda x: sigmoid_(x) * (1 - delta) + 0.5 * delta
sigmoid2 = lambda x: sigmoid(x) * 2.0
logsigmoid = lambda x: -softplus(-x)
logit = lambda x: torch.log
log = lambda x: torch.log(x * 1e2) - np.log(1e2)
logit = lambda x: log(x) - log(1 - x)