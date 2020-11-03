from __future__ import print_function
import argparse
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from scipy.misc import logsumexp
from simple_vae import NonLocalVAE, VAE




def train(epoch, model, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        # loss, BCE = model.calculate_loss(recon_batch, data)
        recon_batch, mu, logvar = model(data)
        loss, BCE = model.calculate_loss(recon_batch, data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        # z_sample = model.z.view(-1, latent_dim).cpu().detach().numpy().mean(axis=0)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, BCE: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data), BCE.item()/data.shape[0]))
            # print(mu.mean().item(), logvar.mean().item())

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))



def test_elbo(x, model, latent_samples = 50):
    model.eval()
    for i in range(latent_samples):
        recon_batch, _, _ = model(x)
        test_loss_tmp, BCE = model.calculate_loss(recon_batch.mean(dim=0), x)
        test_loss_tmp, BCE = test_loss_tmp.view(1), BCE.view(1)

        if i == 0:
            a = test_loss_tmp.cpu().data.numpy()
            b = BCE.cpu().data.numpy()
        else:
            a = np.concatenate((a, test_loss_tmp.cpu().data.numpy()), axis=0)
            b = np.concatenate((b, BCE.cpu().data.numpy()), axis=0)
    elbo_x = logsumexp(-a) - np.log(len(a))
    likelihood_x = logsumexp(-b) - np.log(len(b))
    return likelihood_x, elbo_x  # 1st > 2nd


def test(epoch, model, latent_dim):
    model.eval()
    marginal = 0
    marginal_elbo = 0
    with torch.no_grad():
        if isinstance(model, NonLocalVAE):
            z_sample = model.z.view(-1, latent_dim).cpu().detach().numpy().mean(axis=0)
            print('est z {}'.format(z_sample.round(3)))

        for i, (data, _) in enumerate(test_loader):
            _, mu, logvar = model(data.cuda())
            # print(mu.mean().item(), logvar.mean().item())
            if i > 1:
                continue
            data = data.to(device)
            for j in range(data.shape[0]):
                tmp1, tmp2 = test_elbo(data[j], model, latent_samples=10)
                marginal, marginal_elbo = marginal + tmp1, marginal_elbo + tmp2
            avg_test_BCE = marginal / (data.shape[0] * (i+1))
            avg_marginal_likelihood = marginal_elbo / (data.shape[0] * (i+1))
            print('batch{} avg_test_marginal_BCE {}, neg avg_marginal_likelihood {}'.format(i, avg_test_BCE, avg_marginal_likelihood))


def main(args):
    p = 0.8
    prior = 'imom'  # 'imom' 'Gaussian', vanilla
    tau = 1

    for latent_dim in [50, 100, 200]:
        print('\n CONFIG: latent_dim {}, p {}, prior {} \n'.format(latent_dim, p, prior))
        if prior == 'vanilla':
            model = VAE(latent_dim=latent_dim).to(device)
            p = 1
        else:
            model = NonLocalVAE(latent_dim=latent_dim, prior=prior, tau=tau, p=p).to(device)
        # model = nn.DataParallel(model)
        print(model.parameters)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1, args.epochs + 1):
            t1 = time.time()
            train(epoch, model, optimizer)
            t2 = time.time()
            print('training dur {:.3f}'.format(t2 - t1))
            test(epoch, model, latent_dim)
            t3 = time.time()
            print('testing dur {:.3f}'.format(t3 - t2))

        torch.save(model.state_dict(), args.result_dir + '/nlpvae_{}_latent{}_epoch{}_tau{}_p{}.pt'.format(prior, latent_dim, args.epochs, tau, p))

        # torch.save(model.state_dict(), args.result_dir + '/test.pt')

if __name__ == "__main__":
    import os
    import time

    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument("--result_dir", type=str, default='/extra/yadongl10/git_project/nlpresult/vae_exp/0219')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/extra/yadongl10/data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/extra/yadongl10/data', train=False, transform=transforms.ToTensor()),
        batch_size=100, shuffle=True, **kwargs)

    main(args)


        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
            # save_image(sample.view(64, 1, 28, 28),
            #            'results/sample_' + str(epoch) + '.png')