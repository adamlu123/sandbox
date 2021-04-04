# Author Yadong

# 1. load data: HL shape = 2*num_HL + Pt + Mass + Ntowers + N_q + N_b + target = 22
# 2. models: HLNet
# 3. training
# 4. evaluation
import argparse
import time
import h5py
import numpy as np

parser = argparse.ArgumentParser(description='Multi-prong Model')
parser.add_argument(
    "--strength", type=int, default=0,
    help="regularization strength"
    )
parser.add_argument(
    "--num_hidden", type=int, default=5,
    help="number of latent layer"
    )
parser.add_argument(
    "--inter_dim", type=int, default=800,
    help="hidden layer dimension"
    )
parser.add_argument(
    "--out_dim", type=int, default=64,
    help="output dimension of HLNetBase"
    )
parser.add_argument(
    "--do_rate", type=float, default=4e-1,
    help="dropout rate"
    )
parser.add_argument(
    "--result_dir", type=str,
    default="/baldig/physicsprojects2/N_tagger/exp/archive/test"
    # default="/baldig/physicsprojects2/N_tagger/exp/exp_ptcut/2020308_search_HLnet/HLNet_inter_dim800_num_hidden5_lr1e-4_batch_size256_do3e-1"
    )
parser.add_argument('--model_type', default='HLNet')
parser.add_argument('--stage', default='eval', help='mode in [eval, train]')
parser.add_argument('--load_pretrained', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1000)')
parser.add_argument('--seed', type=int, default=123, help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.001)')
parser.add_argument("--GPU", type=str, default='3', help='GPU id')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
print('training using GPU:', args.GPU)

import torch
import torch.nn as nn
from torch import optim
from resnet import make_hlnet_base
from torch.utils.tensorboard import SummaryWriter

stage = args.stage

writer = SummaryWriter(args.result_dir) if stage == 'train' else None
strength = args.strength
num_labels = 7

## I/O config
device = 'cuda'
lr = args.lr
batchsize = args.batch_size
epoch = args.epochs
result_dir = args.result_dir

## loging:
# ================ HL+mass file
# filename = '/baldig/physicsprojects2/N_tagger/data/merged/parsedTower_res1_res5_merged_mass300_700_b_u_shuffled.h5'
# filename = '/baldig/physicsprojects2/N_tagger/data/v20200302_data/res8_all_HL_target_cutted.h5'
filename = '/baldig/physicsprojects2/N_tagger/data/v20200302_data/merged_res123457910.h5'
# filename = '/baldig/physicsprojects2/N_tagger/data/v20200302_data/merged_N4test.h5'

# ================ target file
fn_target = '/baldig/physicsprojects2/N_tagger/exp/exp_no_ptcut/test/combined_pred_all.h5'
tower_subset = 'parsedTower' # or tower_from_img parsed_Tower
# ================ efps file
dv, nv = 7, 5
fn_efps = '/baldig/physicsprojects2/N_tagger/data/efp/20200319_circularcenter_{}_d{}_n{}/efp_merge.h5'.format('parsed_Tower_cir_centered', dv, nv)
# fn_efps = '/baldig/physicsprojects2/N_tagger/data/efp/20200307_{}_d{}_n{}/efp_merge.h5'.format(tower_subset, dv, nv)
# fn_efps = '/baldig/physicsprojects2/N_tagger/data/efp/20200202_{}_d{}_n{}/efp_merge.h5'.format(tower_subset, dv, nv)


with h5py.File(filename, 'r') as f:
    total_num_sample = f['target'].shape[0]
train_cut, val_cut, test_cut = int(total_num_sample * 0.8), int(total_num_sample * 0.9), total_num_sample

iterations = int(train_cut / batchsize)
iter_test = int((total_num_sample - val_cut) / batchsize)
print('total number samples, train iter and test iter', total_num_sample, iterations, iter_test)


################### data generator
def data_generator(filename, batchsize, start, stop=None, weighted=False):
    iexample = start
    with h5py.File(filename, 'r') as f:
        while True:
            batch = slice(iexample, iexample + batchsize)
            HL_unnorm = f['HL'][batch, :-4]
            # TODO: mass + HL from img --> shape=(-1, 17)
            # HL = f['HL_from_img_normalized'][batch]
            # TODO: mass + HL from tower --> shape=(-1, 17)
            HL = f['HL_normalized'][batch, :-4]
            HL = np.delete(HL, -2, 1)  # delete pt TODO  mass: -1: mass, -2: pt
            # TODO: mass + HL3 from tower --> shape=(-1, 25)
            # HL = f['HL3_normalized'][batch, :-1]
            target = f['target'][batch]
            if weighted:
                weights = f['weights'][batch]
                yield HL, target, weights, HL_unnorm
            else:
                yield HL, target, HL_unnorm
            iexample += batchsize
            if iexample + batchsize >= stop:
                iexample = start


def efp_data_generator(filename, batchsize, start, stop=None):
    iexample = start
    with h5py.File(filename, 'r') as f:
        while True:
            batch = slice(iexample, iexample + batchsize)
            # efps = f['efp_merge'][batch, :]
            efps = f['efp_merge_normalized'][batch, :]
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
generator['train'] = data_generator(filename, batchsize, start=0, stop=train_cut, weighted=False)
generator['val'] = data_generator(filename, batchsize, start=train_cut, stop=val_cut, weighted=False)
generator['test'] = data_generator(filename, batchsize, start=val_cut, stop=test_cut, weighted=False)

efp_generator = {}
efp_generator['train'] = efp_data_generator(fn_efps, batchsize, start=0, stop=train_cut)
efp_generator['val'] = efp_data_generator(fn_efps, batchsize, start=train_cut, stop=val_cut)
efp_generator['test'] = efp_data_generator(fn_efps, batchsize, start=val_cut, stop=test_cut)

target_generator = {}
target_generator['train'] = target_data_generator(fn_target, batchsize, start=0, stop=train_cut)
target_generator['val'] = target_data_generator(fn_target, batchsize, start=train_cut, stop=val_cut)
target_generator['test'] = target_data_generator(fn_target, batchsize, start=val_cut, stop=test_cut)

################### model
hlnet_base = make_hlnet_base(input_dim=17, inter_dim=args.inter_dim, num_hidden=args.num_hidden, out_dim=args.out_dim, do_rate=args.do_rate,batchnorm_base=True) # 25 for HL3 version
efpnet_base = make_hlnet_base(input_dim=567, inter_dim=args.inter_dim, num_hidden=args.num_hidden, out_dim=args.out_dim, do_rate=args.do_rate,batchnorm_base=True)  # 207 566 126


class HLNet(nn.Module):
    def __init__(self, hlnet_base, num_labels=7):
        super(HLNet, self).__init__()
        self.hlnet_base = hlnet_base
        self.top = nn.Linear(64, num_labels)

    def forward(self, HL):
        HL = self.hlnet_base(HL)
        out = self.top(HL)
        return out


class EFPNet(nn.Module):
    def __init__(self, efpnet_base, num_labels=7):
        super(EFPNet, self).__init__()
        self.efpnet_base = efpnet_base
        self.top = nn.Linear(64, num_labels)

    def forward(self, efps):
        efps = self.efpnet_base(efps)
        out = self.top(efps)
        return out


class HLefpNet(nn.Module):
    def __init__(self, hlnet_base, efpnet_base, num_labels=7):
        super(HLefpNet, self).__init__()
        self.hlnet_base = hlnet_base
        self.efpnet_base = efpnet_base
        self.top = nn.Linear(int(hlnet_base.out_dim+efpnet_base.out_dim), num_labels)
        self.bn1 = nn.BatchNorm1d(hlnet_base.out_dim)
        self.bn2 = nn.BatchNorm1d(efpnet_base.out_dim)

    def forward(self, HL, efps):
        HL = self.hlnet_base(HL)
        efps = self.efpnet_base(efps)
        HL, efps = self.bn1(HL), self.bn2(efps)
        out = self.top(torch.cat([HL, efps], dim=-1))
        return out


class GatedHLefpNet(nn.Module):
    def __init__(self, hlnet_base, efpnet_base, num_labels=7):
        super(GatedHLefpNet, self).__init__()
        # self.logit_gates = nn.Parameter(data=torch.zeros(566))
        self.gates = nn.Parameter(data=torch.randn(567))
        self.hlnet_base = hlnet_base
        self.efpnet_base = efpnet_base
        self.top = nn.Linear(128, num_labels)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

    def path(self, HL, efps, gates):
        efps = gates * efps
        HL = self.hlnet_base(HL)
        efps = self.efpnet_base(efps)
        HL, efps = self.bn1(HL), self.bn2(efps)
        out = self.top(torch.cat([HL, efps], dim=-1))
        return out

    def forward(self, HL, efps):
        # efps = torch.sigmoid(self.logit_gates) * efps
        out = self.path(HL, efps, self.gates)
        if model.training:
            return out, self.gates
        else:
            gates_clipped = torch.where(self.gates.abs() > 1e-2, self.gates, torch.zeros_like(self.gates))
            out_clipped = self.path(HL, efps, gates_clipped)
            return out, out_clipped, self.gates


model_type = args.model_type
print('building model:', model_type)
if model_type == 'HLNet':
    model = HLNet(hlnet_base, num_labels=num_labels).to(device)
elif model_type in ['HLefpNet', 'GatedHLefpNet']:
    model = eval(model_type)(hlnet_base, efpnet_base, num_labels=num_labels).to(device)
elif model_type == 'EFPNet':
    model = EFPNet(efpnet_base, num_labels=num_labels).to(device)

model = nn.DataParallel(model)

if args.load_pretrained:
    print('load model from', result_dir)
    cp = torch.load(result_dir + '/best_{}_merged_b_u.pt'.format(model_type)) # '/{}_merged_b_u_ep99.pt'.format(model_type))
    # cp = torch.load(result_dir + '/{}_merged_b_u_ep{}.pt'.format(model_type, 999))
    model.load_state_dict(cp, strict=True)

################### training
hl_param = []
other_param = []
for name, param in model.named_parameters():
    if 'hlnet_base' in name:
        hl_param.append(param)
    else:
        other_param.append(param)

print(model)
print(torch.__version__)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('total#', count_parameters(model), 'num of param in hl', len(hl_param), 'num of param in other', len(other_param))
param_groups = [
    {'params': hl_param, 'lr': lr},
    {'params': other_param, 'lr': lr}
]

optimizer = optim.Adam(param_groups, weight_decay=0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600, 800], gamma=0.5, last_epoch=-1)
loss_fn = nn.CrossEntropyLoss(reduction='none')


def train(model, optimizer, epoch):
    model.train()
    for i in range(iterations):
        optimizer.zero_grad()
        HL, target_long, _ = next(generator['train'])
        target = next(target_generator['train'])
        efps = next(efp_generator['train'])

        HL, target, target_long, efps = torch.tensor(HL).float().to(device), \
                                 torch.tensor(target).float().to(device), torch.tensor(target_long).long().to(device), \
                                                     torch.tensor(efps).float().to(device)

        if model_type == 'HLNet':
            pred = model(HL)
        elif model_type == 'HLefpNet':
            pred = model(HL, efps)
        elif model_type == 'GatedHLefpNet':
            pred, gates = model(HL, efps)
        elif model_type == 'EFPNet':
            pred = model(efps)

        loss = loss_fn(pred, target_long)
        # loss = - (F.log_softmax(target, dim=1).exp() * F.log_softmax(pred, dim=1)).sum(1)
        # loss = ((target - pred) ** 2).sum(1)
        if model_type == 'GatedHLefpNet':
            gate_loss = strength * torch.abs(gates).mean() # (torch.abs(gates) > 1e-2).sum()
            loss = loss.mean() + gate_loss
        else:
            gate_loss = 0.
            loss = loss.mean()
        loss.backward()
        optimizer.step()
        acc = (torch.argmax(pred, dim=1) == target_long).sum().item() / target_long.shape[0]

        if i % 50 == 0:
            print('train loss:', loss.item(), 'gate loss', gate_loss, 'acc:', acc)
            writer.add_scalar('Loss/train', loss.item(), epoch * iterations + i)
            writer.add_scalar('Acc/train', acc, epoch * iterations + i)

    val_acc, val_clipped_acc, _, _, _ = test(model, 'val', epoch)
    return val_acc, val_clipped_acc, model


def test(model, subset, epoch, feature_id=None):
    # torch.manual_seed(123)
    mass_range = [300, 700]
    mass_bins = np.linspace(mass_range[0], mass_range[1], 11)
    bin_len = mass_bins[1] - mass_bins[0]
    pred_mass_list = []
    pred_original_list = []

    model.eval()
    tmp = 0
    tmp_clipped = 0
    if model_type == 'GatedHLefpNet':
        gates = model.module.gates if isinstance(model, nn.DataParallel) else model.gates
    else:
        gates = torch.tensor([0])
    with torch.no_grad():
        for i in range(iter_test):
            HL, target_long, HL_unnorm = next(generator[subset])
            target = next(target_generator[subset])
            efps = next(efp_generator[subset])

            HL_unnorm = torch.tensor(HL_unnorm).float().to(device)
            HL, target, efps = torch.tensor(HL).float().to(device), \
                                          torch.tensor(target).float().to(device), torch.tensor(efps).float().to(device)

            target_long = torch.tensor(target_long).long().to(device)
            pred_armax_clipped = torch.zeros_like(target_long)
            if model_type == 'HLNet':
                if feature_id != None: HL[:, feature_id] = torch.zeros_like(HL[:, feature_id]) #HL[:, feature_id][torch.randperm(batchsize)]  #
                pred = model(HL)
            elif model_type == 'HLefpNet':
                pred = model(HL, efps)
            elif model_type == 'GatedHLefpNet':
                if feature_id: efps[:, feature_id] = efps[:, feature_id][torch.randperm(batchsize)] # torch.zeros_like(efps[:, feature_id])
                pred, pred_clipped, _ = model(HL, efps)
                pred_armax_clipped = torch.argmax(pred_clipped, dim=1)
                tmp_clipped += torch.sum(pred_armax_clipped == target_long).item() / target_long.shape[0]
            elif model_type == 'EFPNet':
                pred = model(efps)
            pred_armax = torch.argmax(pred, dim=1)
            tmp += torch.sum(pred_armax==target_long).item() / target_long.shape[0]

            mass, pt = HL_unnorm[:, -1], HL_unnorm[:, -2]
            bin_idx = torch.floor((mass - mass_range[0]) / bin_len)
            rslt = pred_armax == target_long
            rslt_clipped = pred_armax_clipped == target_long
            pred_mass_list.append(torch.stack([pred_armax.float(), target_long.float(), rslt.float(), rslt_clipped.float(), mass, pt]).cpu())
            pred_original_list.append(pred.cpu())

    clipped_acc = tmp_clipped / iter_test if model_type == 'GatedHLefpNet' else 0.
    gates = gates.detach().cpu().numpy()
    num_remaining_efps = (np.abs(gates) > 1e-2).sum().tolist() if model_type == 'GatedHLefpNet' else 0.

    print(model_type, 'acc', tmp / iter_test, 'clipped acc', clipped_acc, 'num remaining:', num_remaining_efps)
    if stage == 'train':
        print('write to tensorboard ...')
        writer.add_scalar('Acc/val', tmp / iter_test, epoch)
        writer.add_scalar('Acc_clipped/val', clipped_acc, epoch)
        writer.add_scalar('num_remaining_efps', num_remaining_efps, epoch)

    return tmp / iter_test, clipped_acc, pred_original_list, pred_mass_list, gates


def main(model):
    best_acc = 0
    if stage == 'eval':
    # evaluate individual efps
        save_dict = {}
        if model_type == 'GatedHLefpNet':
            gates = model.module.gates.detach().cpu().numpy()
            gates_selected_i = np.where(np.abs(gates) > 1e-2)[0]
            gates_selected_v = gates[gates_selected_i]
            gates_ascend = np.argsort(gates_selected_v)[::-1]
            print('gates>0.01', gates_selected_i)
            num_remaining_efps = len(gates_selected_i) if model_type == 'GatedHLefpNet' else 0.
            print(gates_selected_i[gates_ascend])

            efp_ids = []
            for efp_id in gates_selected_i[gates_ascend]:
                efp_ids.append(efp_id)
                print('testing removing efp_id', efp_id)
                testacc, testclipped_acc, pred_original_list, pred_mass_list, gates = test(model, 'test', epoch=None, feature_id=efp_id)
                save_dict[efp_id] = testclipped_acc

            testacc, testclipped_acc, pred_original_list, pred_mass_list, gates = test(model, 'test', epoch=None)
            save_dict['full'] = testclipped_acc
            np.save('/baldig/physicsprojects2/N_tagger/exp/exp_ptcut/pred/circularcenter_importance_efps_perm_noise_s10.npy', save_dict)
        elif model_type == 'HLNet':
            for HL_id in range(16):
                print('testing removing efp_id', HL_id)
                testacc, testclipped_acc, pred_original_list, pred_mass_list, gates = test(model, 'test', epoch=None,
                                                                                           feature_id=HL_id)
                save_dict[HL_id] = testacc
            testacc, testclipped_acc, pred_original_list, pred_mass_list, gates = test(model, 'test', epoch=None)
            save_dict['full'] = testacc
            # np.save('/baldig/physicsprojects2/N_tagger/exp/exp_ptcut/pred/importance_hl_remove.npy', save_dict)
        # print('saved {} analysis results!'.format(model_type))

    # get combined_pred
    #     gates = model.module.gates.detach().cpu().numpy()
    #     gates_selected_i = np.where(np.abs(gates) > 1e-2)[0]
    #     num_remaining_efps = len(gates_selected_i) if model_type == 'GatedHLefpNet' else 0.

        testacc, testclipped_acc, pred_original_list, pred_mass_list, gates = test(model, 'test', epoch=None)
        combined_pred = torch.cat(pred_mass_list, dim=1).numpy()
        pred_original_list = torch.cat(pred_original_list, dim=0).numpy()
        print('acc', combined_pred[2,:].sum()/combined_pred.shape[1], 'cliped acc', combined_pred[3,:].sum()/combined_pred.shape[1])

        # if model_type == 'GatedHLefpNet':
        #     with h5py.File('/baldig/physicsprojects2/N_tagger/exp/exp_ptcut/pred/combined_pred_efps567_inter_dim800_num_hidden5_do3e_1_corrected.h5', 'a') as f:
        #         f.create_dataset('circularcenter_savewoclip_{}_strength{}_best'.format(model_type, strength), data=combined_pred)
        #         f.create_dataset('circularcenter_savewoclip_{}_strength{}_best_original'.format(model_type, strength), data=pred_original_list)
        #         f.create_dataset('circularcenter_savewoclip_{}_strength{}_best_n_remain'.format(model_type, strength), data=num_remaining_efps)
        #         f.create_dataset('circularcenter_savewoclip_{}_strength{}_gates'.format(model_type, strength), data=gates)
        # else:
        #     with h5py.File('/baldig/physicsprojects2/N_tagger/exp/exp_ptcut/pred/combined_pred_all_N4test.h5', 'a') as f:
        #         # if '{}_best'.format(model_type) in f:
        #         #     del f['{}_best'.format(model_type)]
        #         f.create_dataset('circularcenter_{}_best'.format(model_type), data=combined_pred)
        #         f.create_dataset('circularcenter_{}_best_original'.format(model_type), data=pred_original_list)
        # print('saving finished!')

    else:
        for i in range(epoch):
            print('starting epoch', i)
            val_acc, val_clipped_acc, model = train(model, optimizer, epoch=i)
            val_metric = val_clipped_acc if args.model_type == 'GatedHLefpNet' else val_acc
            if val_metric > best_acc:
                best_acc = val_clipped_acc
                torch.save(model.state_dict(), result_dir + '/best_{}_merged_b_u.pt'.format(model_type))
                print('model saved')
            if (i+1) % 50 == 0:
                torch.save(model.state_dict(),
                           result_dir + '/{}_merged_b_u_ep{}.pt'.format(model_type, i))
                print('model saved at epoch', i)
        writer.close()
        # testacc = test(model, 'test')
        # print('test acc', testacc)


if __name__ == '__main__':
    main(model)



# from shutil import copyfile
# root = '/baldig/physicsprojects2/N_tagger/exp/efps'
# exp_name = '/2020201_lr_5e-3_decay0.5_nowc_weighted_sample_corrected_image_efp'
# exp_name = '/2020207_lr_5e-3_decay0.5_nowc_weighted_sample_corrected_image_noramed_efp_hl_original'
# exp_name = '/2020201_lr_5e-3_decay0.5_nowc_weighted_sample_corrected_image_normed_efp_only_d7_n5_parsed_tower'
#     os.makedirs(root + exp_name)
#     copyfile('/extra/yadongl10/git_project/sandbox/multi_prongs/efp_exp.py', root + exp_name + '/efp_exp.py')