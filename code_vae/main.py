from __future__ import print_function
import argparse
import numpy as np
import h5py
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch.utils.data
from tqdm import tqdm
import time
from torch import nn, optim
from torch.nn import functional as F
import pickle as pkl


def load_data(subset='727Methods'):
    data_dir = '/extra/yadongl10/data'
    start = time.time()
    with h5py.File(data_dir+'/java.h5', 'r') as f:
        datasets = np.asarray(f[subset][:]) # 10
    print('finish load data in {:5f} sec'.format(time.time()-start))
    return datasets


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2110, out_channels=100, kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=10, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(490, 20)
        self.fc2 = nn.Linear(490, 20)

    def forward(self, x):
        x = x.view(-1, 2110, 200)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        return self.fc1(x), self.fc2(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(20, 2110)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=50, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=50, out_channels=100, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=100, out_channels=200, kernel_size=3, stride=1, padding=1)
        # self.fc4 = nn.Linear(400, 2110)

    def forward(self, z):
        z = F.relu(self.fc1(z)).view(-1, 1, 2110)
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))

        return z


class Conv_VAE(nn.Module):
    def __init__(self):
        super(Conv_VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, data):
        mu, logvar = self.encoder(data)
        latent = self.reparameterize(mu, logvar)
        recon_batch = self.decoder(latent)
        return recon_batch, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = -torch.sum(x * F.log_softmax(recon_x, dim=2), dim=(1,2))  # (5, 200, 2110) -> (5)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (5,20) -> (5)

    return (BCE + KLD).mean(dim=0)



def train(data, model, optimizer, epoch, batchsize):
    model.train()
    num_steps = data.shape[0]//batchsize
    pbar = tqdm(total=num_steps)

    for batch_idx in range(num_steps):
        optimizer.zero_grad()
        batch_data = data[batch_idx*batchsize:(batch_idx+1)*batchsize]
        recon_batch, mu, logvar = model(batch_data)
        loss = loss_function(recon_batch, batch_data, mu, logvar).mean()

        loss.backward()
        optimizer.step()
    pbar.update(batchsize)
    pbar.set_description('train, loss {:.5f}'.format(loss.item()))
    return model


def plot(model, data):
    model.eval()
    mu, logvar = model.encoder(data)
    z = model.reparameterize(mu, logvar)
    z = z.detach().cpu().numpy()

    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(z)
    print(X_reduced.shape)
    # (N, 2)
    # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
    # for i, c in enumerate(colors):
    #     plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=c, label=str(i))

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='b')
    plt.legend()
    plt.savefig('tsne.png', dpi=200)


def main(args):
    model = Conv_VAE().to(device)
    data = load_data(subset='727Methods')
    data = torch.tensor(data, dtype=torch.float).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(args.epochs):
        model = train(data, model, optimizer, epoch, args.batch_size)
        print('loss after epoch {}'.format(epoch))
    if args.save_model:
        torch.save(model.state_dict(), 'conv_vae0.pt')
        print('model saved!')

    plot(model, data)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Code VAE')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = 'cuda'  # torch.device("cuda" if args.cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    main(args)

