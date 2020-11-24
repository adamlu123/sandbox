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
from resnet import resnet20base, make_hlnet_base, resnet110base
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

## load config
device = 'cuda'
batchsize = 128
epoch = 100
# filename = '/extra/yadongl10/data/N-tagger/combined_1103.h5'
# total_num_sample = 215630
# filename = '/baldig/physicsprojects2/N_tagger/20201109/combined_1109.h5'
# filename = '/baldig/physicsprojects2/N_tagger/res3/res3_mass300_700_b_u_shuffled.h5'
# filename = '/baldig/physicsprojects2/N_tagger/merged/merged_mass300_700_b_u_shuffled.h5'
filename = '/baldig/physicsprojects2/N_tagger/merged/res1_res5_merged_mass300_700_b_u_shuffled.h5'

with h5py.File(filename, 'r') as f:
    total_num_sample = f['image'].shape[0]
train_cut, val_cut, test_cut = int(total_num_sample * 0.8), int(total_num_sample * 0.9), total_num_sample

iterations = int(train_cut / batchsize)
iter_test = int((val_cut - train_cut) / batchsize)
print('total number samples, train iter and test iter', total_num_sample, iterations, iter_test)


################### data generator
def data_generator(filename, batchsize, start, stop=None, weighted=False):
    iexample = start
    with h5py.File(filename, 'r') as f:
        while True:
            batch = slice(iexample, iexample + batchsize)
            X = f['image'][batch, :, :]
            HL = f['HL_normalized'][batch, :-4]
            target = f['target'][batch]
            if weighted:
                weights = f['weights'][batch]
                yield X, HL, target, weights
            else:
                yield X, HL, target
            iexample += batchsize
            if iexample + batchsize >= stop:
                iexample = start

generator = {}
generator['train'] = data_generator(filename, batchsize, start=0, stop=train_cut)
generator['val'] = data_generator(filename, batchsize, start=train_cut, stop=val_cut)
generator['test'] = data_generator(filename, batchsize, start=val_cut, stop=test_cut)


################### model
resnetbase = resnet110base()
hlnet_base = make_hlnet_base()


class JetImageNet(nn.Module):
    def __init__(self, resnetbase):
        super(JetImageNet, self).__init__()
        self.resnetbase = resnetbase
        self.top = nn.Linear(64, 6)

    def forward(self, X):
        X = X.unsqueeze(1)
        X = self.resnetbase(X)
        out = self.top(X)
        return out


class JetImageMassNet(nn.Module):
    def __init__(self, resnetbase):
        super(JetImageMassNet, self).__init__()
        self.resnetbase = resnetbase
        self.top = nn.Linear(64*2, 6)
        self.m1 = nn.Linear(1, 10)
        self.m2 = nn.Linear(10, 64)

    def forward(self, X, mass):
        X = X.unsqueeze(1)
        X = self.resnetbase(X)
        m = self.m1(mass.view(-1, 1))
        m = self.m2(m)
        out = self.top(torch.cat([X, m], dim=-1))
        return out


class HLNet(nn.Module):
    def __init__(self, hlnet_base):
        super(HLNet, self).__init__()
        self.hlnet_base = hlnet_base
        self.top = nn.Linear(64, 6)

    def forward(self, HL):
        HL = self.hlnet_base(HL)
        out = self.top(HL)
        return out


class CombinedNet(nn.Module):
    def __init__(self, resnetbase, hlnet_base):
        super(CombinedNet, self).__init__()
        self.resnetbase = resnetbase
        self.hlnet_base = hlnet_base
        self.top = nn.Linear(64*2, 6)

    def forward(self, X, HL):
        HL = self.hlnet_base(HL)
        X = X.unsqueeze(1)
        X = self.resnetbase(X)
        out = self.top(torch.cat([HL, X], dim=-1))
        return out


model_type = 'CombinedNet'
if model_type == 'JetImageNet':
    model = JetImageNet(resnetbase).to(device)
elif model_type == 'CombinedNet':
    model = CombinedNet(resnetbase, hlnet_base).to(device)
elif model_type == 'HLNet':
    model = HLNet(hlnet_base).to(device)
elif model_type == 'JetImageMassNet':
    model = JetImageMassNet(resnetbase).to(device)

model = nn.DataParallel(model)

################### training
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(reduction='none')


def train(model, optimizer):
    model.train()
    for i in range(iterations):
        optimizer.zero_grad()
        # X, HL, target, weights = next(generator['train'])
        # X, HL, target, weights = torch.tensor(X).float().to(device), torch.tensor(HL).float().to(device), \
        #                          torch.tensor(target).long().to(device), torch.tensor(weights).to(device)
        X, HL, target = next(generator['train'])
        X, HL, target = torch.tensor(X).float().to(device), torch.tensor(HL).float().to(device), \
                                 torch.tensor(target).long().to(device)
        if model_type == 'JetImageNet':
            pred = model(X)
        elif model_type == 'CombinedNet':
            pred = model(X, HL)
        elif model_type == 'HLNet':
            pred = model(HL)
        elif model_type == 'JetImageMassNet':
            pred = model(X, HL[:, -1])
        loss = loss_fn(pred, target) #* weights * 10
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        acc = (torch.argmax(pred, dim=1) == target).sum().item() / target.shape[0]

        if i % 50 == 0:
            print('train loss:', loss.item(), 'acc:', acc)
        # if i % 500 == 0 and i != 0:
        #     print(loss.item())
    val_acc = test(model, 'val')
    return val_acc, model


def test(model, subset):
    # model.eval()
    tmp = 0
    with torch.no_grad():
        for i in range(iter_test):
            # X, HL, target, _ = next(generator[subset])
            # X, HL, target = torch.tensor(X).float().to(device), torch.tensor(HL).float().to(device), torch.tensor(target).to(device)
            X, HL, target = next(generator[subset])
            X, HL, target = torch.tensor(X).float().to(device), torch.tensor(HL).float().to(device), torch.tensor(target).to(device)
            if model_type == 'JetImageNet':
                pred = model(X)
            elif model_type == 'CombinedNet':
                pred = model(X, HL)
            elif model_type == 'HLNet':
                pred = model(HL)
            elif model_type == 'JetImageMassNet':
                pred = model(X, HL[:, -1])
            pred = torch.argmax(pred, dim=1)
            tmp += torch.sum(pred==target).item() / target.shape[0]
    print(model_type, 'acc', tmp / iter_test)
    return tmp / iter_test


def main(model):
    best_acc = 0
    for i in range(epoch):
        print('starting epoch', i)
        val_acc, model = train(model, optimizer)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), '/baldig/physicsprojects2/N_tagger/exp/20201124/best_{}_merged_b_u.pt'.format(model_type))
            print('model saved')
        # if (i+1) % 10 == 0:
        #     torch.save(model.state_dict(),
        #                '/baldig/physicsprojects2/N_tagger/exp/20201119/{}_merged_b_u_ep{}.pt'.format(model_type, i))
        #     print('model saved at epoch', i)
        #     test(model, 'test')
    testacc = test(model, 'test')
    print('test acc', testacc)


if __name__ == '__main__':
    main(model)