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