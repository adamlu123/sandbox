# standard library imports
from __future__ import absolute_import, division, print_function
import argparse

parser = argparse.ArgumentParser(description='Sparse Auto-regressive Model')
parser.add_argument(
    "--num_hidden", type=int, default=3,
    help="number of latent layer"
    )
parser.add_argument(
    "--psize", type=int, default=100,
    help="hidden layer dimension"
)
parser.add_argument(
    "--fsize", type=int, default=100,
    help="hidden layer dimension"
    )
parser.add_argument(
    "--result_dir", type=str,
    default=None
    )
parser.add_argument('--stage', default='eval', help='mode in [eval, train]')
parser.add_argument('--load_pretrained', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1000)')
parser.add_argument('--seed', type=int, default=123, help='random seed (default: 1)')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
parser.add_argument("--GPU", type=str, default='0', help='GPU id')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
print('training using GPU:', args.GPU)


# standard numerical library imports
import numpy as np
import tensorflow as tf
# tf.config.gpu.set_per_process_memory_fraction(0.75)
# tf.config.gpu.set_per_process_memory_growth(True)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


import energyflow as ef
from energyflow.archs import PFN
from energyflow.utils import data_split, remap_pids, to_categorical
import h5py
filename = '/baldig/physicsprojects2/N_tagger/merged/parsedTower_res1_res5_merged_mass300_700_b_u_shuffled.h5'
with h5py.File(filename, 'r') as f:
    X = np.array(f['parsed_Tower'])
    y = np.array(f['target'])
    HL = np.array(f['HL_normalized'])[:, :-4]
    HL_unnorm = np.array(f['HL'])[:, :-4]
mass = HL_unnorm[:, -1]
HL_unnorm.shape, HL.shape

# convert labels to categorical
Y = to_categorical(y, num_classes=6)

# do train/val/test split
total_num_sample = Y.shape[0]

# center + normalize the tower
for x in X:
    mask = x[:, 0] > 0
    yphi_avg = np.average(x[mask, 1:3], weights=x[mask, 0], axis=0)
    x[mask, 1:3] -= yphi_avg
    x[mask, 0] /= x[:, 0].sum()

train_cut, val_cut, test_cut = int(total_num_sample * 0.8), int(total_num_sample * 0.9), total_num_sample
X_train, X_val, X_test = X[:train_cut], X[train_cut:val_cut], X[val_cut:]
Y_train, Y_val, Y_test = Y[:train_cut], Y[train_cut:val_cut], Y[val_cut:]
mass_test = mass[val_cut:]
print('Done train/val/test split', X_train.shape, X_val.shape, X_test.shape)

# build architecture
num_hidden = args.num_hidden
psize, fsize = args.psize, args.fsize

Phi_sizes, F_sizes = (psize,)*num_hidden, (fsize,)*num_hidden
batch_size = args.batch_size
print(Phi_sizes)
pfn = PFN(input_dim=X.shape[-1], output_dim=6, Phi_sizes=Phi_sizes, F_sizes=F_sizes)

                                                                   #num_hidden{}_psize{}_fsize{}_batchsize{}_ep{}
log_dir="/baldig/physicsprojects2/N_tagger/exp/20210223_PFN_search/num_hidden{}_psize{}_fsize{}_batchsize{}_ep{}".format(num_hidden, psize, fsize, batch_size, args.epochs)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.result_dir, histogram_freq=1)

# train model
pfn.fit(X_train, Y_train,
        epochs=args.epochs,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        verbose=1,
        callbacks=[tensorboard_callback])


# save model
# pfn.save("/baldig/physicsprojects2/N_tagger/exp/20210223_PFN_search/PFN_psize{}_fsize{}_ep{}".format(psize, fsize, args.epochs))