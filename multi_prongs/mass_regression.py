# Author

# 1. load data
# 2. models: HLNet, JetImageNet, CombinedNet
# 3. training
# 4. evaluation

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from resnet import resnet20base
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

## load config
device = 'cuda'
batchsize = 128
epoch = 100
# filename = '/extra/yadongl10/data/N-tagger/combined_1103.h5'
# total_num_sample = 215630
filename = '/baldig/physicsprojects2/N_tagger/20201109/combined_1109.h5'
total_num_sample = 375842
train_cut, val_cut, test_cut = int(total_num_sample * 0.8), int(total_num_sample * 0.9), total_num_sample
iterations = int(train_cut / batchsize)
iter_test = int((val_cut - train_cut) / batchsize)
print(iterations, iter_test)

################### data generator
def data_generator(filename, batchsize, start, stop=None):
    iexample = start
    with h5py.File(filename, 'r') as f:
        while True:
            batch = slice(iexample, iexample + batchsize)
            X = f['image'][batch, :, :]
            HL = f['HL'][batch, :]
            target = f['trasformed_target'][batch]
            yield X, HL, target
            iexample += batchsize
            if iexample + batchsize >= stop:
                iexample = start

generator = {}
generator['train'] = data_generator(filename, batchsize, start=0, stop=train_cut)
generator['val'] = data_generator(filename, batchsize, start=train_cut, stop=val_cut)
generator['test'] = data_generator(filename, batchsize, start=val_cut, stop=test_cut)


################### model
resnetbase = resnet20base()


class JetImageNet(nn.Module):
    def __init__(self, resnetbase):
        super(JetImageNet, self).__init__()
        self.resnetbase = resnetbase
        self.top = nn.Linear(64, 1)

    def forward(self, X):
        X = X.unsqueeze(1)
        X = self.resnetbase(X)
        out = self.top(X)
        return out

model = JetImageNet(resnetbase).to(device)

################### training
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


def train(model, optimizer):
    model.train()
    loss_fn = nn.MSELoss()
    for i in range(iterations):
        optimizer.zero_grad()
        X, HL, target = next(generator['train'])
        X, HL, target = torch.tensor(X).float().to(device), torch.tensor(HL).float().to(device), torch.tensor(target).long().to(device)
        mass_target = (HL[:, -5] - 241.21) / 160.58
        pred = model(X).squeeze()
        # pred = torch.relu(pred)
        loss = ((pred - mass_target)**2).mean()
        # loss = loss_fn(pred, mass_target)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(loss.item())
        if i % 500 == 0 and i != 0:
            print(loss.item())
            test(model, 'val')
    return


def test(model, subset):
    # model.eval()
    tmp = 0
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for i in range(iter_test):
            X, HL, target = next(generator[subset])
            X, HL, target = torch.tensor(X).float().to(device), torch.tensor(HL).float().to(device), torch.tensor(target).to(device)
            mass_target = (HL[:, -5] - 241.21) / 160.58
            pred = model(X).squeeze()
            # pred = torch.relu(pred)
            loss = ((pred - mass_target)**2).mean()
            # loss = loss_fn(pred, mass_target)
            tmp += loss.item()
    print('acc', tmp / iter_test)


def main():
    for i in range(epoch):
        train(model, optimizer)
    test(model, 'test')


if __name__ == '__main__':
    main()