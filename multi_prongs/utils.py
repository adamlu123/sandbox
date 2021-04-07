import numpy as np
from torch.utils.data import Dataset


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


def split_data(X, Y, HL, fold_id, num_folds=10):
    total_num_sample = X.shape[0]
    X, Y = cross_validate(X, Y, fold_id=fold_id, num_folds=num_folds)
    train_cut, val_cut, test_cut = int(total_num_sample * 0.8), int(total_num_sample * 0.9), total_num_sample

    X_train, X_val, X_test = X[:train_cut], X[train_cut:val_cut], X[val_cut:]
    Y_train, Y_val, Y_test = Y[:train_cut], Y[train_cut:val_cut], Y[val_cut:]
    HL_train, HL_val, HL_test = HL[:train_cut], HL[train_cut:val_cut], HL[val_cut:]

    data_train = MethodDataset(X_train, Y_train, HL_train)
    data_val = MethodDataset(X_val, Y_val, HL_val)
    data_test = MethodDataset(X_test, Y_test, HL_test)

    return data_train, data_val, data_test


class MethodDataset(Dataset):
    def __init__(self, X, Y, HL):
        self.X, self.Y, self.HL = X, Y, HL
        self.n = X.shape[0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.HL[index]


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