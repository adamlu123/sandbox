import os
import sys
import copy
import math
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, f, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(1)
    x = x.permute(0, 2, 1)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    num_dims, num_dims_f = x.shape[1], f.shape[-1]

    # x = x.transpose(2,
    #                 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = f.reshape(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims_f)
    # x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    f = f.view(batch_size, num_points, 1, num_dims_f).repeat(1, 1, k, 1)

    feature = torch.cat((feature - f, f), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(nn.Module):
    def __init__(self, emb_dims=64, k=10, dropout=0.5, output_channels=6):
        super(DGCNN, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        # input x shape:(batch, num_points, 3)
        batch_size = x.size(0)
        pts, fts = x[:, :, 1:], x[:, :, 0].unsqueeze(-1)
        x = get_graph_feature(pts, fts, k=self.k)  # (128, 2, 230, 10)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0].permute(0, 2, 1)  # (128, 230, 64)

        x = get_graph_feature(x1, x1, k=self.k)     # shape (128, 128, 230, 10)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0].permute(0, 2, 1)  # (128, 230, 64)

        x = get_graph_feature(x2, x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0].permute(0, 2, 1)

        x = get_graph_feature(x3, x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0].permute(0, 2, 1)

        x = torch.cat((x1, x2, x3, x4), dim=-1)

        x = self.conv5(x.permute(0, 2, 1))
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


# class ParticleNet(nn.Module):
#     def __init__(self, DGCNN):
#         super(ParticleNet, self).__init__()
#         self.dgcnn = DGCNN
#     def forward(self, *input):
#         feature =


## load config
device = 'cuda'
batchsize = 128
epoch = 100
load_pretrained = True
root = '/baldig/physicsprojects2/N_tagger/exp'
exp_name = '/20201211_lr_5e-3_decay0.5_nowc_bert_toweronly'
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
generator['train'] = data_generator(filename, batchsize, start=0, stop=train_cut, weighted=False)
generator['val'] = data_generator(filename, batchsize, start=train_cut, stop=val_cut, weighted=False)
generator['test'] = data_generator(filename, batchsize, start=val_cut, stop=test_cut, weighted=False)



################### model
model_type = 'DGCNN'
if model_type == 'DGCNN':
    model = DGCNN(output_channels=6).to(device)

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
        if model_type == 'DGCNN':
            pred = model(X)
        loss = loss_fn(pred, target)  # * weights
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
            if model_type == 'DGCNN':
                pred = model(X)
            pred = torch.argmax(pred, dim=1)
            tmp += torch.sum(pred == target).item() / target.shape[0]
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
        if (i + 1) % 10 == 0:
            # torch.save(model.state_dict(),
            #            root + exp_name + '/{}_merged_b_u_ep{}.pt'.format(model_type, i))
            print('model saved at epoch', i)
        #     test(model, 'test')
    testacc = test(model, 'test')
    print('test acc', testacc)


if __name__ == '__main__':
    main(model)