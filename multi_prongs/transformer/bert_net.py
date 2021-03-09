from transformers import *
import argparse

parser = argparse.ArgumentParser(description='Sparse Auto-regressive Model')
parser.add_argument(
    "--num_hidden", type=int, default=4,
    help="number of latent layer"
    )
parser.add_argument(
    "--hidden_size", type=int, default=128,
    help="embedding size"
    )
parser.add_argument(
    "--inter_dim", type=int, default=128,
    help="intermediate layer size"
    )
parser.add_argument(
    "--num_attention_heads", type=int, default=8,
    help="num of attention heads"
    )
parser.add_argument(
    "--result_dir", type=str,
    default="/baldig/physicsprojects2/N_tagger/exp/exp_no_ptcut/efps/20200209_HLNet_inter_dim800_num_hidden5"
    )
parser.add_argument('--stage', default='train', help='mode in [eval, train]')
parser.add_argument('--model_type', default='bert')
parser.add_argument('--load_pretrained', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1000)')
parser.add_argument('--seed', type=int, default=123, help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
parser.add_argument("--GPU", type=str, default='2', help='GPU id')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
print('training using GPU:', args.GPU)
print(str(args))

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from bert import BertForSequenceClassification
from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter

## load config
device = 'cuda'
batchsize = args.batch_size
epoch = args.epochs
load_pretrained = True
# root = '/baldig/physicsprojects2/N_tagger/exp/exp_ptcut'
# exp_name = '/2020308_lr_1e-4_decay0.5_nowc_bertmass_tower_from_img_embed512_hidden6_head8'
result_dir = args.result_dir

stage = args.stage
if stage != 'eval':
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    writer = SummaryWriter(result_dir) if stage != 'eval' else None
    ## loging:
    copyfile('/extra/yadongl10/git_project/sandbox/multi_prongs/transformer/bert_net.py', result_dir + '/bert_net.py')
    # filename = '/baldig/physicsprojects2/N_tagger/data/merged/parsedTower_res1_res5_merged_mass300_700_b_u_shuffled.h5'
    filename = '/baldig/physicsprojects2/N_tagger/data/v20200302_data/merged_res123457910.h5'

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
            # X = f['tower_from_img'][batch, :, :]
            X = f['parsed_Tower_centered'][batch, :, :]
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
class TransformerClassification(nn.Module):
    def __init__(self, d_model, nhead, dropout, num_layers, seq_len=230):
        super(TransformerClassification, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed = nn.Linear(3, d_model, bias=False)
        self.pool = nn.Linear(d_model, 1)
        self.out_linear = nn.Linear(seq_len, 6)

    def forward(self, X):
        src = self.embed(X)    # batch, S, embed_size
        src_key_padding_mask = torch.where(X[:, :, 0] != 0, torch.ones_like(X[:, :, 0]),
                                           torch.zeros_like(X[:, :, 0]))
        out = self.transformer_encoder(src=src.permute([1,0,2]), src_key_padding_mask=src_key_padding_mask.bool())
        out = out.permute([1,0,2])  # batch, S, embed_size
        out = self.pool(out).squeeze()
        out = self.out_linear(out)
        return out


class BertMassNet(nn.Module):
    def __init__(self, bert_model):
        super(BertMassNet, self).__init__()
        self.bert = bert_model
        self.top = nn.Linear(6*2, 6)
        self.m1 = nn.Linear(1, 6)

    def forward(self, X, mass):
        X = self.bert(X)
        m = self.m1(mass.reshape(-1, 1))
        out = self.top(torch.cat([X, m], dim=-1))
        return out


config = BertConfig(
                    hidden_size=args.hidden_size,  #256,
                    num_hidden_layers=args.num_hidden, num_attention_heads=args.num_attention_heads,
                    intermediate_size=args.inter_dim, num_labels=7,
                    input_dim=230,
                    attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1
                    )

model_type = 'bert'
if model_type == 'bert':
    model = BertForSequenceClassification(config).to(device)
elif model_type == 'BertMassNet':
    bert_model = BertForSequenceClassification(config)
    model = BertMassNet(bert_model).to(device)
elif model_type == 'transformer':
    model = TransformerClassification(d_model=16, nhead=8, dropout=0.1, num_layers=6).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('training parameters', count_parameters(model))
model = nn.DataParallel(model)

################### training
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
# optimizer = Lamb(model.parameters(), lr=1e-3, weight_decay=0, adam=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600, 800], gamma=0.5, last_epoch=-1)
loss_fn = nn.CrossEntropyLoss(reduction='none')


def train(model, optimizer, epoch):
    model.train()
    for i in range(iterations):
        optimizer.zero_grad()
        # X, HL, target, weights = next(generator['train'])
        # X, HL, target, weights = torch.tensor(X).float().to(device), torch.tensor(HL).float().to(device), \
        #                          torch.tensor(target).long().to(device), torch.tensor(weights).to(device)
        X, HL, target = next(generator['train'])
        X, target = torch.tensor(X).float().to(device), \
                        torch.tensor(target).long().to(device)
        if model_type == 'bert' or model_type == 'transformer':
            pred = model(X)
        elif model_type == 'BertMassNet':
            HL = torch.tensor(HL).float().to(device)
            pred = model(X, HL[:, -1])
        loss = loss_fn(pred, target)  # * weights
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        acc = (torch.argmax(pred, dim=1) == target).sum().item() / target.shape[0]

        if i % 50 == 0:
            print('train loss:', loss.item(), 'acc:', acc)
            writer.add_scalar('Loss/train', loss.item(), epoch * iterations + i)
            writer.add_scalar('Acc/train', acc, epoch * iterations + i)
    val_acc = test(model, 'val', epoch)
    return val_acc, model


def test(model, subset, epoch):
    model.eval()
    tmp = 0
    with torch.no_grad():
        for i in range(iter_test):
            # X, HL, target, _ = next(generator[subset])
            # X, HL, target = torch.tensor(X).float().to(device), torch.tensor(HL).float().to(device), torch.tensor(target).to(device)
            X, HL, target = next(generator[subset])
            X, target = torch.tensor(X).float().to(device), torch.tensor(target).to(device)
            if model_type == 'bert' or model_type == 'transformer':
                pred = model(X)
            elif model_type == 'BertMassNet':
                HL = torch.tensor(HL).float().to(device)
                pred = model(X, HL[:, -1])
            pred = torch.argmax(pred, dim=1)
            tmp += torch.sum(pred == target).item() / target.shape[0]
    print(model_type, 'acc', tmp / iter_test)
    writer.add_scalar('Acc/val', tmp / iter_test, epoch)
    return tmp / iter_test


def main(model):
    best_acc = 0
    for i in range(epoch):
        print('starting epoch', i)
        val_acc, model = train(model, optimizer, i)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), result_dir + '/best_{}_merged_b_u.pt'.format(model_type))
            print('model saved')
        if (i + 1) % 10 == 0:
            # torch.save(model.state_dict(),
            #            result_dir + '/{}_merged_b_u_ep{}.pt'.format(model_type, i))
            print('model saved at epoch', i)
        #     test(model, 'test')
    testacc = test(model, 'test', i)
    print('test acc', testacc)


if __name__ == '__main__':
    main(model)