# standard library imports
from __future__ import absolute_import, division, print_function
import argparse
# import sys
# sys.path.append('/extra/yadongl10/git_project/sandbox/multi_prongs')
# from utils import cross_validate
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

parser = argparse.ArgumentParser(description='Sparse Auto-regressive Model')
parser.add_argument(
    "--num_hidden", type=int, default=4,
    help="number of latent layer"
    )
parser.add_argument(
    "--psize", type=int, default=256,
    help="hidden layer dimension"
)
parser.add_argument(
    "--fsize", type=int, default=256,
    help="hidden layer dimension"
    )
parser.add_argument(
    "--dropout", type=float, default=25e-2,
    help="dropout ratio"
    )
parser.add_argument(
    "--result_dir", type=str,
    default='/baldig/physicsprojects2/N_tagger/exp/exp_ptcut/20210309_PFN_search_batch256/do2e-1_num_hidden4_psize256_fsize256_batchsize256_ep1000'
    )
parser.add_argument('--stage', default='eval', help='mode in [eval, train]')
parser.add_argument('--load_pretrained', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 100)')
parser.add_argument('--fold_id', type=int, default=0, help='CV fold in [0, 9]')
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
# import energyflow as ef
from energyflow.archs import PFN
from energyflow.utils import data_split, remap_pids, to_categorical
import h5py
num_class = 7
# filename = '/baldig/physicsprojects2/N_tagger/data/merged/parsedTower_res1_res5_merged_mass300_700_b_u_shuffled.h5'
# filename = '/baldig/physicsprojects2/N_tagger/data/v20200302_data/merged_N4test.h5'
filename = '/baldig/physicsprojects2/N_tagger/data/v20200302_data/merged_res123457910.h5'
print('----------------------------------------')
print('cross validation: fold id ', args.fold_id, filename)
print('----------------------------------------')


with h5py.File(filename, 'r') as f:
    X = np.array(f['parsed_Tower_eta_arith_phi_cir_centered']) # [parsed_Tower_eta_arith_phi_cir_centered parsed_Tower_cir_centered]
    y = np.array(f['target'])
    # HL = np.array(f['HL_normalized'])[:, :-4]
    HL_unnorm = np.array(f['HL'])[:, :-4]
# with h5py.File('/baldig/physicsprojects2/N_tagger/data/v20200302_data/merged_res1234579_v2_hl.h5', 'r') as f:
#     HL = np.array(f['HL_normalized'][:, :-4])
mass, pt = HL_unnorm[:, -1], HL_unnorm[:, -2]
# convert labels to categorical
Y = to_categorical(y, num_classes=num_class)

# do train/val/test split
total_num_sample = Y.shape[0]

# center + normalize the tower
# for x in X:
#     mask = x[:, 0] > 0
#     yphi_avg = np.average(x[mask, 1:3], weights=x[mask, 0], axis=0)
#     x[mask, 1:3] -= yphi_avg
#     x[mask, 0] /= x[:, 0].sum()


X, Y = cross_validate(X, Y, fold_id=args.fold_id, num_folds=10)
train_cut, val_cut, test_cut = int(total_num_sample * 0.8), int(total_num_sample * 0.9), total_num_sample
X_train, X_val, X_test = X[:train_cut], X[train_cut:val_cut], X[val_cut:]
Y_train, Y_val, Y_test = Y[:train_cut], Y[train_cut:val_cut], Y[val_cut:]
print('Done train/val/test split', X_train.shape, X_val.shape, X_test.shape)
print(args.result_dir)
# build architecture
num_hidden = args.num_hidden
psize, fsize = args.psize, args.fsize

# Phi_sizes, F_sizes = (128, ) * 2, (fsize,)*num_hidden # (psize,)*num_hidden,
Phi_sizes, F_sizes = (psize,)*num_hidden, (fsize,)*num_hidden #

batch_size = args.batch_size
opt = tf.keras.optimizers.Adam(lr=args.lr)
compile_opts = {'optimizer': opt}

print(Phi_sizes, 'lr', args.lr)
# pfn = PFN(input_dim=X.shape[-1], output_dim=6, Phi_sizes=Phi_sizes, F_sizes=F_sizes, compile_opts=compile_opts)
pfn = PFN(input_dim=X.shape[-1], output_dim=num_class, Phi_sizes=Phi_sizes, F_sizes=F_sizes, F_dropouts=args.dropout, compile_opts=compile_opts)


if args.stage == 'train':
    if args.load_pretrained:
        pfn.load_weights(args.result_dir + '/best')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.result_dir, histogram_freq=0)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.result_dir + '/best',
        save_weights_only=True,
        monitor='val_acc',
        mode='max',
        save_best_only=True)

    # train model
    pfn.fit(X_train, Y_train,
            epochs=args.epochs,
            batch_size=batch_size,
            validation_data=(X_val, Y_val),
            verbose=1,
            callbacks=[tensorboard_callback, model_checkpoint_callback])

elif args.stage == 'eval':
    print('load model from', args.result_dir)
    pfn.load_weights(args.result_dir + '/best')

    preds = pfn.predict(X_test, batch_size=256)
    preds_argmax = np.argmax(preds, 1)
    target = np.argmax(Y_test, 1)
    rslt = preds_argmax == target
    print(preds_argmax.shape, 'acc', np.sum(rslt / Y_test.shape[0]))
    mass, pt = mass[val_cut:], pt[val_cut:]
    combined_pred = np.stack([preds_argmax, target, rslt, mass, pt])
    print(combined_pred.shape)
    # with h5py.File('/baldig/physicsprojects2/N_tagger/exp/exp_ptcut/pred/cross_valid/combined_pred_all_cv_correctetacenter.h5', 'a') as f:
    #     # del f['circularcenter_{}'.format('PFN')]
    #     # del f['circularcenter_{}_original'.format('PFN')]
    #     f.create_dataset('fold{}_{}_best'.format(args.fold_id, 'PFN'), data=combined_pred)
    #     f.create_dataset('fold{}_{}_best_original'.format(args.fold_id, 'PFN'), data=preds)
    # print('saving finished!')
else:
    raise ValueError('only support stage in: [train, eval]!')

# save model
# pfn.save(args.result_dir)
