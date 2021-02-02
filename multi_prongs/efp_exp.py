# Author Yadong

# 1. load data
# 2. models: HLNet
# 3. training
# 4. evaluation

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1, 2"
from shutil import copyfile
from resnet import make_hlnet_base


## load config
device = 'cuda'
batchsize = 128
epoch = 500
load_pretrained = True
root = '/baldig/physicsprojects2/N_tagger/exp/efps'
exp_name = '/2020201_lr_5e-3_decay0.5_nowc_weighted_sample_corrected_image_normed_efp_hl_original'
if not os.path.exists(root + exp_name):
    os.makedirs(root + exp_name)
## loging:
copyfile('/extra/yadongl10/git_project/sandbox/multi_prongs/efp_exp.py', root+exp_name+'/efp_exp.py')
filename = '/baldig/physicsprojects2/N_tagger/merged/parsedTower_res1_res5_merged_mass300_700_b_u_shuffled.h5'
fn_efps = '/baldig/physicsprojects2/N_tagger/efp/20200201_tower_original/efp_merge.h5' # 20200201_tower_original 20200201_tower_img
fn_target = '/baldig/physicsprojects2/N_tagger/exp/test/combined_pred_all.h5'

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
            X = f['image_corrected'][batch, :, :]
            # HL = f['HL_from_img_normalized'][batch]
            HL = f['HL_normalized'][batch, :-4]
            HL = np.delete(HL, -2, 1)  # delete pt TODO  mass: -1: mass, -2: pt
            HL = HL / HL.std(0)
            target = f['target'][batch]
            if weighted:
                weights = f['weights'][batch]
                yield X, HL, target, weights
            else:
                yield X, HL, target
            iexample += batchsize
            if iexample + batchsize >= stop:
                iexample = start


def efp_data_generator(filename, batchsize, start, stop=None):
    iexample = start
    with h5py.File(filename, 'r') as f:
        while True:
            batch = slice(iexample, iexample + batchsize)
            efps = f['efp_merge'][batch, :]
            efps = (efps - efps.mean(axis=0)) / efps.std(axis=0)
            yield efps
            iexample += batchsize
            if iexample + batchsize >= stop:
                iexample = start


def target_data_generator(filename, batchsize, start, stop=None):
    iexample = start
    with h5py.File(filename, 'r') as f:
        while True:
            batch = slice(iexample, iexample + batchsize)
            # pred_mass_list: (3,num_samples) consists of: pred, target, bin_idx, binary rslt (true/false)
            pred = f['bert_predonly'][batch, :]  # JetImageMassNet_predonly, bert_predonly
            yield pred
            iexample += batchsize
            if iexample + batchsize >= stop:
                iexample = start


generator = {}
generator['train'] = data_generator(filename, batchsize, start=0, stop=train_cut, weighted=True)
generator['val'] = data_generator(filename, batchsize, start=train_cut, stop=val_cut, weighted=True)
generator['test'] = data_generator(filename, batchsize, start=val_cut, stop=test_cut, weighted=True)

efp_generator = {}
efp_generator['train'] = efp_data_generator(fn_efps, batchsize, start=0, stop=train_cut)
efp_generator['val'] = efp_data_generator(fn_efps, batchsize, start=train_cut, stop=val_cut)
efp_generator['test'] = efp_data_generator(fn_efps, batchsize, start=val_cut, stop=test_cut)

target_generator = {}
target_generator['train'] = target_data_generator(fn_target, batchsize, start=0, stop=train_cut)
target_generator['val'] = target_data_generator(fn_target, batchsize, start=train_cut, stop=val_cut)
target_generator['test'] = target_data_generator(fn_target, batchsize, start=val_cut, stop=test_cut)

################### model
hlnet_base = make_hlnet_base(input_dim=17)
efpnet_base = make_hlnet_base(input_dim=207)


class HLNet(nn.Module):
    def __init__(self, hlnet_base):
        super(HLNet, self).__init__()
        self.hlnet_base = hlnet_base
        self.top = nn.Linear(64, 6)

    def forward(self, HL):
        HL = self.hlnet_base(HL)
        out = self.top(HL)
        return out


class HLefpNet(nn.Module):
    def __init__(self, hlnet_base, efpnet_base):
        super(HLefpNet, self).__init__()
        self.hlnet_base = hlnet_base
        self.efpnet_base = efpnet_base
        self.top = nn.Linear(128, 6)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, HL, efps):
        HL = self.hlnet_base(HL)
        efps = self.efpnet_base(efps)
        HL, efps = self.bn1(HL), self.bn2(efps)
        out = self.top(torch.cat([HL, efps], dim=-1))
        return out


model_type = 'HLefpNet'
if model_type == 'HLNet':
    model = HLNet(hlnet_base).to(device)
elif model_type == 'HLefpNet':
    model = HLefpNet(hlnet_base, efpnet_base).to(device)

model = nn.DataParallel(model)

if load_pretrained:
    cp = torch.load(root + exp_name + '/best_{}_merged_b_u.pt'.format(model_type))
    # cp = torch.load('/baldig/physicsprojects2/N_tagger/exp/20201203_lr_5e-3_decay0.5_nowc_weighted_sample_corrected_image/best_{}_merged_b_u.pt'.format(model_type))
    model.load_state_dict(cp, strict=False)

################### training
hl_param = []
other_param = []
for name, param in model.named_parameters():
    if 'hlnet_base' in name:
        hl_param.append(param)
    else:
        other_param.append(param)
print('num of param in hl', len(hl_param), 'num of param in other', len(other_param))
param_groups = [
    {'params': hl_param, 'lr': 5e-3},
    {'params': other_param, 'lr': 5e-3}
]

optimizer = optim.Adam(param_groups, weight_decay=0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5, last_epoch=-1)
loss_fn = nn.CrossEntropyLoss(reduction='none')


def train(model, optimizer):
    model.train()
    for i in range(iterations):
        optimizer.zero_grad()
        X, HL, target_long, weights = next(generator['train'])
        target = next(target_generator['train'])
        efps = next(efp_generator['train'])
        X, HL, target, target_long, weights, efps = torch.tensor(X).float().to(device), torch.tensor(HL).float().to(device), \
                                 torch.tensor(target).float().to(device), torch.tensor(target_long).long().to(device), \
                                                    torch.tensor(weights).to(device), torch.tensor(efps).float().to(device)

        if model_type == 'HLNet':
            pred = model(HL)
        elif model_type == 'HLefpNet':
            pred = model(HL, efps)

        # loss = loss_fn(pred, target)
        # loss = ((target - pred) ** 2).sum(1)
        loss = - (F.log_softmax(target, dim=1).exp() * F.log_softmax(pred, dim=1)).sum(1)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        acc = (torch.argmax(pred, dim=1) == target_long).sum().item() / target.shape[0]

        if i % 50 == 0:
            print('train loss:', loss.item(), 'acc:', acc)
    val_acc = test(model, 'val')
    return val_acc, model


def test(model, subset):
    model.eval()
    tmp = 0
    with torch.no_grad():
        for i in range(iter_test):
            X, HL, target_long, weights = next(generator[subset])
            target = next(target_generator[subset])
            efps = next(efp_generator[subset])
            X, HL, target, weights, efps = torch.tensor(X).float().to(device), torch.tensor(HL).float().to(device), \
                                          torch.tensor(target).float().to(device), torch.tensor(weights).to(device), \
                                          torch.tensor(efps).float().to(device)
            target_long = torch.tensor(target_long).long().to(device)
            if model_type == 'HLNet':
                pred = model(HL)
            elif model_type == 'HLefpNet':
                pred = model(HL, efps)
            pred = torch.argmax(pred, dim=1)
            tmp += torch.sum(pred==target_long).item() / target.shape[0]
    print(model_type, 'acc', tmp / iter_test)
    return tmp / iter_test


def main(model):
    best_acc = 0
    # testacc = test(model, 'test')
    for i in range(epoch):
        print('starting epoch', i)
        val_acc, model = train(model, optimizer)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), root + exp_name + '/best_{}_merged_b_u.pt'.format(model_type))
            print('model saved')
        if (i+1) % 10 == 0:
            torch.save(model.state_dict(),
                       root + exp_name + '/{}_merged_b_u_ep{}.pt'.format(model_type, i))
            print('model saved at epoch', i)
    testacc = test(model, 'test')
    print('test acc', testacc)


if __name__ == '__main__':
    main(model)