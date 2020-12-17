from transformers import *

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import os
from bert import BertForSequenceClassification

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
config = BertConfig(
                    hidden_size=3,  # 25
                    num_hidden_layers=10, num_attention_heads=3,
                    intermediate_size=10, num_labels=6,
                    input_dim=230
                    )

model_type = 'bert'
if model_type == 'bert':
    model = BertForSequenceClassification(config).to(device)

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
        if model_type == 'bert':
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
            X, HL, target = torch.tensor(X).float().to(device), torch.tensor(HL).float().to(device), torch.tensor(
                target).to(device)
            if model_type == 'bert':
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