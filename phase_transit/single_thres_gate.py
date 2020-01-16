import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import pickle as pkl
from tqdm import tqdm
import torch.utils.data

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
        X = MultivariateNormal(loc=torch.zeros(self.n), covariance_matrix=torch.eye(self.n)).sample(
            torch.Size([self.K]))
        self.X, self.Y = X, Y

    def __len__(self):
        'Denotes the total number of samples'
        return self.K

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def get_data_loader(q, K, n, m, transform=True):
    dataset = Dataset(q, K, n, m, transform=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
    return data_loader


def get_acc(label, pred):
    pred = torch.where(pred < 0, -torch.ones_like(pred), torch.ones_like(pred))
    return (label == pred).sum().float() / label.shape[0]


class SingleThresholdGate(nn.Module):
    def __init__(self, n, m):
        super(SingleThresholdGate, self).__init__()
        self.linear = nn.Linear(in_features=n, out_features=m, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return torch.tanh(x)
        # return torch.where(x<0.5, torch.zeros_like(x), torch.ones_like(x))



def train(data_loader, model, optimizer):
    pbar = tqdm(total=len(data_loader.dataset))
    train_loss = 0

    for batch_idx, (data, label) in enumerate(data_loader):
        data.to(device)
        optimizer.zero_grad()

        pred = model(data).squeeze()
        loss = max_margin(label, pred)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        pbar.update(data.size(0))
        pbar.set_description('train, loss {:.5f}'.format(train_loss/(batch_idx + 1)))
    acc = get_acc(label, pred)

    return acc


def main(epochs, q, K, n, m=1, epsilon=1e-5):
    data_loader = get_data_loader(q, K, n, m)
    print('finish getting data!')
    model = SingleThresholdGate(n, m)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    acc = []
    for epoch in range(epochs):
        acc.append(train(data_loader, model, optimizer))
        print('acc at end of epoch {}: {:.5f}'.format(epoch, acc[-1]))
    acc = torch.tensor(acc[-10:]).numpy()
    return acc





if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Flow spike and slab')
    parser.add_argument('--C', type=int, default=10)
    parser.add_argument('--result_dir', type=str, default='/extra/yadongl10/BIG_sandbox/NN_Capacity/phase_transit/01152020')
    # parser.add_argument('--q', type=float, default=1/(20*np.log(10)), help='y prob')
    # parser.add_argument('--K', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--m', type=int, default=1)
    args = parser.parse_args()

    K = args.C * args.n
    q = 1/(2*args.C * np.log(args.C))

    results = []
    for prob in [0.5*q, q, 5*q, 10*q]:
        results.append(main(args.epochs, prob, K, args.n, args.m, epsilon=1e-5))
    out = np.asarray(results)

    with open(args.result_dir + '/acc.pkl', 'wb') as f:
        pkl.dump(out, f)











