from transformers import *
import argparse
import sys
sys.path.append('/extra/yadongl10/git_project/sandbox/multi_prongs')
from utils import cross_validate, get_id_range

parser = argparse.ArgumentParser(description='Sparse Auto-regressive Model')
parser.add_argument(
    "--num_hidden", type=int, default=4,
    help="number of latent layer"
    )
parser.add_argument(
    "--hidden_size", type=int, default=512,
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
    default='/baldig/physicsprojects2/N_tagger/exp/archive/2020307_lr_1e-4_decay0.5_nowc_bertmass_tower_from_img_embed512_hidden4_head8'
    # default="/baldig/physicsprojects2/N_tagger/exp/exp_ptcut/2020308_lr_1e-4_decay0.5_nowc_bertmass_tower_from_img_embed512_hidden6_head8"
    )
parser.add_argument('--stage', default='eval', help='mode in [eval, train]')
parser.add_argument('--model_type', default='bert')
parser.add_argument('--load_pretrained', action='store_true', default=False)
parser.add_argument('--fold_id', type=int, default=0, help='CV fold in [0, 9]')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1000)')
parser.add_argument('--seed', type=int, default=123, help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
parser.add_argument("--GPU", type=str, default='2', help='GPU id')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU # '0,1,2,3' #
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
result_dir = args.result_dir

stage = args.stage
if stage == 'train':
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    writer = SummaryWriter(result_dir)
    ## loging:
    copyfile('/extra/yadongl10/git_project/sandbox/multi_prongs/transformer/bert_net.py', result_dir + '/bert_net.py')
elif stage == 'eval':
    args.load_pretrained = True

# filename = '/baldig/physicsprojects2/N_tagger/data/merged/parsedTower_res1_res5_merged_mass300_700_b_u_shuffled.h5'
filename = '/baldig/physicsprojects2/N_tagger/data/v20200302_data/merged_res123457910.h5'
# filename = '/baldig/physicsprojects2/N_tagger/data/v20200302_data/merged_N4test.h5'


with h5py.File(filename, 'r') as f:
    total_num_sample = f['target'].shape[0]
train_cut, val_cut, test_cut = int(total_num_sample * 0.8), int(total_num_sample * 0.9), total_num_sample

iterations = int(train_cut / batchsize)
iter_test = int((val_cut - train_cut) / batchsize)
print('total number samples, train iter and test iter', total_num_sample, iterations, iter_test)

train_range, val_range, test_range = get_id_range(total_num_sample, fold_id=args.fold_id, num_folds=10)


################### data generator
def data_generator(filename, batchsize, idx_range, weighted=False):
    start, stop = idx_range
    iexample = start
    with h5py.File(filename, 'r') as f:
        while True:
            batch = slice(iexample, iexample + batchsize)
            # X = f['tower_from_img'][batch, :, :]
            X = f['parsed_Tower_cir_centered'][batch, :, :]
            HL_unnorm = f['HL'][batch, :-4]
            target = f['target'][batch]
            if weighted:
                weights = f['weights'][batch]
                yield X, HL_unnorm, target, weights
            else:
                yield X, HL_unnorm, target
            iexample += batchsize
            if iexample + batchsize >= stop:
                iexample = start

generator = {}
generator['train'] = data_generator(filename, batchsize, train_range, weighted=False)
generator['val'] = data_generator(filename, batchsize, val_range, weighted=False)
generator['test'] = data_generator(filename, batchsize, test_range, weighted=False)


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
                    hidden_size=args.hidden_size,
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

if args.load_pretrained:
    print('load model from', result_dir)
    cp = torch.load(result_dir + '/best_{}_merged_b_u.pt'.format(model_type))
    model.load_state_dict(cp, strict=True)

################### training
if args.stage == 'train':
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
        X, HL_unnorm, target = next(generator['train'])
        X, target = torch.tensor(X).float().to(device), \
                        torch.tensor(target).long().to(device)
        if model_type == 'bert' or model_type == 'transformer':
            pred = model(X)
        elif model_type == 'BertMassNet':
            HL_unnorm = torch.tensor(HL_unnorm).float().to(device)
            pred = model(X, HL_unnorm[:, -1])
        loss = loss_fn(pred, target)  # * weights
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        acc = (torch.argmax(pred, dim=1) == target).sum().item() / target.shape[0]

        if i % 50 == 0:
            print('train loss:', loss.item(), 'acc:', acc)
            writer.add_scalar('Loss/train', loss.item(), epoch * iterations + i)
            writer.add_scalar('Acc/train', acc, epoch * iterations + i)
    val_acc, _ = test(model, 'val', epoch)
    return val_acc, model


def test(model, subset, epoch):
    model.eval()
    tmp = 0
    pred_mass_list = []
    pred_original_list = []
    with torch.no_grad():
        for i in range(iter_test):
            # X, HL, target, _ = next(generator[subset])
            # X, HL, target = torch.tensor(X).float().to(device), torch.tensor(HL).float().to(device), torch.tensor(target).to(device)
            X, HL_unnorm, target = next(generator[subset])
            X, target = torch.tensor(X).float().to(device), torch.tensor(target)
            if model_type == 'bert' or model_type == 'transformer':
                pred = model(X)
            elif model_type == 'BertMassNet':
                HL_unnorm = torch.tensor(HL_unnorm).float().to(device)
                pred = model(X, HL_unnorm[:, -1])
            pred_armax = torch.argmax(pred, dim=1).cpu()
            tmp += torch.sum(pred_armax == target).item() / target.shape[0]

            mass, pt = torch.tensor(HL_unnorm[:, -1]), torch.tensor(HL_unnorm[:, -2])
            rslt = pred_armax == target
            pred_mass_list.append(torch.stack([pred_armax.float(), target.float(), rslt.float(), mass.float(), pt.float()]))
            pred_original_list.append(pred.cpu())

    print(model_type, 'acc', tmp / iter_test)
    if stage != 'eval':
        writer.add_scalar('Acc/val', tmp / iter_test, epoch)
    return tmp / iter_test, pred_mass_list, pred_original_list


def main(model):
    best_acc = 0
    if stage == 'eval':
        testacc, pred_mass_list, pred_original_list = test(model, 'test', None)
        combined_pred = torch.cat(pred_mass_list, dim=1).numpy()
        pred_original_list = torch.cat(pred_original_list, dim=0).numpy()
        with h5py.File('/baldig/physicsprojects2/N_tagger/exp/exp_ptcut/pred/combined_pred_all_N4test.h5', 'a') as f:
            f.create_dataset('circularcenter_{}_best_original'.format(model_type), data=pred_original_list)
            if '{}_best'.format(model_type) in f:
                del f['{}_best'.format(model_type)]
            f.create_dataset('circularcenter_{}_best'.format(model_type), data=combined_pred)
        print('saving finished!')

    elif stage == 'train':
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
        testacc, _ = test(model, 'test', i)
        print('test acc', testacc)
    else:
        raise ValueError('only support stage in: [train, eval]!')


if __name__ == '__main__':
    main(model)