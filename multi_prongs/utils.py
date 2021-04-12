import numpy as np
from torch.utils.data import Dataset
import torch


def cross_validate(X, Y, fold_id, num_folds=10):
    """
    :param X:
    :param Y:
    :param fold_id: starts from 1, 2, ..., num_folds
    :param num_folds:
    :return:
    """
    n = X.shape[0]
    test_size = int(n // num_folds)
    test_ids = (fold_id * test_size + np.arange(test_size)).tolist()
    perm = np.array([i for i in range(n) if i not in test_ids] + test_ids)
    return X[perm], Y[perm]


def split_data(X, Y, HL, fold_id=None, num_folds=10):
    total_num_sample = X.shape[0]
    if fold_id is not None:
        X, Y = cross_validate(X, Y, fold_id=fold_id, num_folds=num_folds)
    train_cut, val_cut, test_cut = int(total_num_sample * 0.8), int(total_num_sample * 0.9), total_num_sample

    X_train, X_val, X_test = X[:train_cut], X[train_cut:val_cut], X[val_cut:]
    Y_train, Y_val, Y_test = Y[:train_cut], Y[train_cut:val_cut], Y[val_cut:]
    HL_train, HL_val, HL_test = HL[:train_cut], HL[train_cut:val_cut], HL[val_cut:]

    data_train = MethodDataset(X_train, Y_train, HL_train)
    data_val = MethodDataset(X_val, Y_val, HL_val)
    data_test = MethodDataset(X_test, Y_test, HL_test)

    return data_train, data_val, data_test


def split_data_HL_efp(efp, Y, HL, HL_unnorm, fold_id=None, num_folds=10):
    total_num_sample = HL.shape[0]
    if fold_id is not None:
        X, Y = cross_validate(HL, Y, fold_id=fold_id, num_folds=num_folds)
    train_cut, val_cut, test_cut = int(total_num_sample * 0.8), int(total_num_sample * 0.9), total_num_sample

    efp_train, efp_val, efp_test = efp[:train_cut], efp[train_cut:val_cut], efp[val_cut:]
    Y_train, Y_val, Y_test = Y[:train_cut], Y[train_cut:val_cut], Y[val_cut:]
    HL_train, HL_val, HL_test = HL[:train_cut], HL[train_cut:val_cut], HL[val_cut:]
    HL_unnorm_train, HL_unnorm_val, HL_unnorm_test = HL_unnorm[:train_cut], HL_unnorm[train_cut:val_cut], HL_unnorm[val_cut:]

    data_train = EFPDataset(efp_train, Y_train, HL_train, HL_unnorm_train)
    data_val = EFPDataset(efp_val, Y_val, HL_val, HL_unnorm_val)
    data_test = EFPDataset(efp_test, Y_test, HL_test, HL_unnorm_test)

    return data_train, data_val, data_test


class MethodDataset(Dataset):
    def __init__(self, X, Y, HL):
        self.X, self.Y, self.HL = X, Y, HL
        self.n = X.shape[0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.HL[index]


class EFPDataset(Dataset):
    def __init__(self, X, Y, HL, HL_unorm):
        self.X, self.Y, self.HL, self.HL_unorm = X, Y, HL, HL_unorm
        self.n = X.shape[0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.HL[index], self.HL_unorm[index]


def get_batch_classwise_acc(pred_mass_list):
    combined_pred = torch.cat(pred_mass_list, dim=1).numpy()
    print(combined_pred.shape)
    pred, target = combined_pred[0, :], combined_pred[1, :]
    classwise_acc = []
    for i in range(7):
        tmp = pred[target==i]
        correct = np.where(tmp==i)[0].shape[0]
        # print(correct, tmp.shape[0])
        classwise_acc.append(correct / tmp.shape[0])
    print(classwise_acc)
    return classwise_acc





# def data_generator(filename, batchsize, idx_range, weighted=False):
#     start, stop = idx_range
#     iexample = start
#     with h5py.File(filename, 'r') as f:
#         while True:
#             batch = slice(iexample, iexample + batchsize)
#             # X = f['tower_from_img'][batch, :, :]
#             X = f['parsed_Tower_cir_centered'][batch, :, :]
#             HL_unnorm = f['HL'][batch, :-4]
#             target = f['target'][batch]
#             if weighted:
#                 weights = f['weights'][batch]
#                 yield X, HL_unnorm, target, weights
#             else:
#                 yield X, HL_unnorm, target
#             iexample += batchsize
#             if iexample + batchsize >= stop:
#                 iexample = start

# def get_id_range(n, num_folds, fold_id):
#     test_size = int(n // num_folds)
#     test_ids = (fold_id * test_size + np.arange(test_size)).tolist()
#     perm = np.array([i for i in range(n) if i not in test_ids] + test_ids)
#     train_cut, val_cut, test_cut = int(n * 0.8), int(n * 0.9), n
#     train_id, val_id, test_id = perm[:train_cut], perm[train_cut:val_cut], perm[val_cut:test_cut]
#     return train_id, val_id, test_id