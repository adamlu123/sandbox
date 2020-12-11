
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
import os
from resnet import make_hlnet_base

## load config
device = 'cuda'
batchsize = 128
epoch = 100
load_pretrained = True
root = '/baldig/physicsprojects2/N_tagger/exp'
exp_name = '/20201130_lr_5e-3_decay0.5_nowc_pointconv1d'
if not os.path.exists(root + exp_name):
    os.makedirs(root + exp_name)
filename = '/baldig/physicsprojects2/N_tagger/merged/parsedTower_res1_res5_merged_mass300_700_b_u_shuffled.h5'

with h5py.File(filename, 'r') as f:
    total_num_sample = f['target'].shape[0]
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
            X = f['parsed_Tower'][batch, :, :]
            HL = f['HL'][batch, :-4]
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
generator['train'] = data_generator(filename, batchsize, start=0, stop=train_cut, weighted=False)
generator['val'] = data_generator(filename, batchsize, start=train_cut, stop=val_cut, weighted=False)
generator['test'] = data_generator(filename, batchsize, start=val_cut, stop=test_cut, weighted=False)


################### model
class RNN(nn.Module):
    def __init__(self, rnn):
        super(RNN, self).__init__()
        self.rnn = rnn
        self.module = nn.Linear(230, 6)

    def forward(self, X):
        self.rnn.flatten_parameters()
        X = self.rnn(X.permute([1,0,2]))
        X = F.relu(X[0])
        X = X.permute([1,0,2]).mean(-1)
        out = self.module(X)
        return out


class Conv1DNet(nn.Module):
    def __init__(self):
        super(Conv1DNet, self).__init__()
        modules = [nn.Conv1d(in_channels=2, out_channels=16, kernel_size=1), nn.ReLU(), nn.BatchNorm1d(16),
                   nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1), nn.ReLU(), nn.BatchNorm1d(32),
                   nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1), nn.ReLU()]
        self.module = nn.Sequential(*modules)
        self.linear_xy = nn.Linear(230, 64)
        self.linear_pt = nn.Linear(230, 64)
        self.out = nn.Linear(128, 64)

    def forward(self, X):
        X = X.permute([0, 2, 1])  # shape = batch, 3, 230
        pt, xy = X[:, 0, :], X[:, 1:, :]
        X = self.module(xy).squeeze()
        pt = F.relu(self.linear_pt(pt))
        X = F.relu(self.linear_xy(X))
        X = F.relu(self.out(torch.cat([pt, X], -1)))
        return X


class PointConvNet(nn.Module):
    def __init__(self, hlnet_base, conv1d_base):
        super(PointConvNet, self).__init__()
        self.hlnet_base = hlnet_base
        self.conv1d_base = conv1d_base

         
        # self.top = nn.Linear(64 * 2, 64)
        self.out = nn.Linear(64, 64)

    def forward(self, X, HL):
        HL = self.hlnet_base(HL)
        X = self.conv1d_base(X)
        # out = self.top(torch.cat([HL, X], dim=-1))
        out = self.out(F.relu(X))
        return out


model_type = 'PointConvNet'
if model_type == 'BiLSTM':
    lstm = nn.LSTM(input_size=3, hidden_size=20, num_layers=2, bidirectional=True)
    model = RNN(lstm).to(device)
elif model_type == 'PointConvNet':
    hlnet_base, conv1d_base = make_hlnet_base(), Conv1DNet()
    model = PointConvNet(hlnet_base, conv1d_base).to(device)
model = nn.DataParallel(model)


################### training

optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5, last_epoch=-1)
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
        if model_type == 'BiLSTM':
            pred = model(X)
        elif model_type == 'PointConvNet':
            pred = model(X, HL)
        loss = loss_fn(pred, target) #* weights
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        acc = (torch.argmax(pred, dim=1) == target).sum().item() / target.shape[0]

        if i % 50 == 0:
            print('train loss:', loss.item(), 'acc:', acc)
    val_acc = test(model, 'val')
    return val_acc, model


def test(model, subset):
    model.eval()
    tmp = 0
    with torch.no_grad():
        for i in range(iter_test):
            # X, HL, target, _ = next(generator[subset])
            # X, HL, target = torch.tensor(X).float().to(device), torch.tensor(HL).float().to(device), torch.tensor(target).to(device)
            X, HL, target = next(generator[subset])
            X, HL, target = torch.tensor(X).float().to(device), torch.tensor(HL).float().to(device), torch.tensor(target).to(device)
            if model_type == 'BiLSTM':
                pred = model(X)
            elif model_type == 'PointConvNet':
                pred = model(X, HL)
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
            # torch.save(model.state_dict(), root + exp_name + '/best_{}_merged_b_u.pt'.format(model_type))
            # print('model saved')
        if (i+1) % 10 == 0:
            # torch.save(model.state_dict(),
            #            root + exp_name + '/{}_merged_b_u_ep{}.pt'.format(model_type, i))
            print('model saved at epoch', i)
        #     test(model, 'test')
    testacc = test(model, 'test')
    print('test acc', testacc)


if __name__ == '__main__':
    main(model)