import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import pickle as pkl
from tqdm import tqdm
import torch.utils.data
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = 'cuda'


def max_margin(label, pred):
    return torch.clamp(1 - label * pred, min=0).mean()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, q, K, n, m, transform=True):
        self.q = q
        self.K = K
        self.n = n
        self.m = m
        self.transform = transform
        Y = Bernoulli(probs=self.q).sample(torch.Size([self.K, self.m])).squeeze()
        if self.transform:
            Y = 2 * Y - 1
        # X = MultivariateNormal(loc=torch.zeros(self.n), covariance_matrix=torch.eye(self.n)).sample(
        #     torch.Size([self.K]))
        X = Normal(loc=0., scale=1.).sample(torch.Size([self.K, self.n]))
        self.X, self.Y = X, Y

    def __len__(self):
        'Denotes the total number of samples'
        return self.K

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def get_data_loader(q, K, n, m, batch_size=100):
    dataset = Dataset(q, K, n, m, transform=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def get_acc(model, data_loader):
    label = data_loader.dataset.Y
    pred = model(data_loader.dataset.X)
    pred = torch.where(pred < 0, -torch.ones_like(pred), torch.ones_like(pred)).squeeze()
    return (label == pred).sum().float() / label.shape[0]


class SingleThresholdGate(nn.Module):
    def __init__(self, n, m):
        super(SingleThresholdGate, self).__init__()
        self.linear = nn.Linear(in_features=n, out_features=m, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return torch.tanh(x)
        # return torch.where(x<0.5, torch.zeros_like(x), torch.ones_like(x))



def train(data_loader, model, optimizer, scheduler):
    # pbar = tqdm(total=len(data_loader.dataset))
    train_loss = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data.to(device)
        optimizer.zero_grad()

        pred = model(data).squeeze()
        loss = max_margin(label, pred)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
    scheduler.step()
        # pbar.update(data.size(0))
        # pbar.set_description('train, loss {:.5f}'.format(train_loss/(batch_idx + 1)))

    acc = get_acc(model, data_loader)

    return acc


def main(epochs, q, K, n, m=1, epsilon=1e-5, batch_size=100):
    repeat = 5
    acc_repeat = []  # a list with length repeat
    torch.manual_seed(1)
    for i in range(repeat):

        data_loader = get_data_loader(q, K, n, m, batch_size=batch_size)
        print('finish getting data!')
        model = SingleThresholdGate(n, m)
        # optimizer = optim.Adam(model.parameters(), lr=1e-2)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.98)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs//10),
                                                               eta_min=1e-5, last_epoch=-1)

        acc = []
        for epoch in range(epochs):
            epoch_acc = train(data_loader, model, optimizer, scheduler)
            if epoch_acc == 1:
                break
            acc.append(epoch_acc)
            if epoch % (epochs//10) == 0:
                print('acc at epoch {} is :{:5f}'.format(epoch, epoch_acc.item()))
            # print('acc at end of epoch {}: {:.5f}'.format(epoch, acc[-1]))
        acc = torch.tensor(acc).numpy().max()
        print('acc at of repeat {}: {:.5f}'.format(i, acc))
        acc_repeat.append(acc)

    return acc_repeat





if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Flow spike and slab')
    parser.add_argument('--C', type=int, default=5)
    parser.add_argument('--result_dir', type=str, default='/extra/yadongl10/BIG_sandbox/NN_Capacity/phase_transit/01172020')
    # parser.add_argument('--q', type=float, default=1/(20*np.log(10)), help='y prob')
    # parser.add_argument('--K', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 1000)')
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--m', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1000)

    args = parser.parse_args()

    K = args.C * args.n
    q = 1/(2*args.C * np.log(args.C))

    results = []
    for prob in [0.1*q, 0.2*q, 0.25*q, 0.5*q, q, 1.5*q, 2*q, 2.5*q, 5*q, 10*q, 15*q]:  # [0.5*q, q, 2*q, 5*q]
        print('CONFIG: n:{}; C: {}; transitq = {}; prob: {} * q'.format(args.n, args.C, q, prob/q))
        results.append(main(args.epochs, prob, K, args.n, args.m, epsilon=1e-5, batch_size=K//10))
    out = np.asarray(results)

    with open(args.result_dir + '/acc.pkl', 'wb') as f:
        pkl.dump(out, f)











