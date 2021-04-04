import numpy as np
import torch
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


def get_id_range(n, num_folds, fold_id):
    test_size = int(n // num_folds)
    test_ids = (fold_id * test_size + np.arange(test_size)).tolist()
    perm = np.array([i for i in range(n) if i not in test_ids] + test_ids)
    train_cut, val_cut, test_cut = int(n * 0.8), int(n * 0.9), n
    train_id, val_id, test_id = perm[:train_cut], perm[train_cut:val_cut], perm[val_cut:test_cut]
    return [train_id[0], train_id[-1]], [val_id[0], val_id[-1]], [test_id[0], test_id[-1]]


class methodDataset(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
        self.input_dim = body.shape[1] - 1
        self.type = type

    def __len__(self):
        return len(self.body)

    def __getitem__(self, index):
        loc_vec = torch.zeros(self.input_dim)
        body, exception = self.body[index][:-1], self.exception[index]
        start, end = self.loc[index][0], self.loc[index][1]
        assert end <= self.input_dim
        if self.type == 'train':
            if end > start:
                loc_vec[start:(end+1)] = torch.ones(end - start + 1) / (end+1-start)  # small modification May 4th
            else:
                if start < 0:   # [start, end] = [-1, -1]
                    loc_vec = torch.zeros_like(loc_vec) #torch.ones_like(loc_vec) / self.input_dim
                else:   # start == end
                    loc_vec[start] = 1.
        return body, loc_vec, exception