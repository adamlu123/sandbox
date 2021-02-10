import numpy as np
import pandas as pd
import h5py
import energyflow as ef
import h5py
import time
import os

# cd /extra/yadongl10/git_project/sandbox/multi_prongs/data_preproc python generate_efps.py
# source activate tf180
phase = 'merge'
subset = 'parsed_Tower' # or parsed_Tower

dv, nv = 10, 5
save_dir = '/baldig/physicsprojects2/N_tagger/efp/20200202_{}_d{}_n{}'.format(subset, dv, nv)
print('merge', phase, 'config', "d<{}".format(dv), "n<{}".format(nv), "p==1")

filename = '/baldig/physicsprojects2/N_tagger/merged/parsedTower_res1_res5_merged_mass300_700_b_u_shuffled.h5'
with h5py.File(filename, 'r') as f:
    tower = np.array(f[subset])
print(tower.shape, type(tower))

nonzero_tower = [tower[i][np.where(tower[i, :, 0]>0)] for i in range(tower.shape[0])]
print(len(nonzero_tower))



# Grab graphs
kappas = [-1,0,1]
betas = [1/2, 1, 2]
efpset = ef.EFPSet("d<{}".format(dv), "n<{}".format(nv), "p==1")
graphs = efpset.graphs()

if phase == 'generate':
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for efp_ix, graph in enumerate(graphs):
        n, e, d, v, k, c, p, h = efpset.specs[efp_ix]
        print('config', "d<{}".format(dv), "n<{}".format(nv), "p==1")
        print('efp_ix', efp_ix, 'total', len(graphs), 'ndk', n, d, k)
        s = time.time()
        for kappa in kappas:
            for beta in betas:
                EFP_graph = ef.EFP(
                    graph, measure="hadr", kappa=kappa, beta=beta, normed=True
                )
                efp = EFP_graph.batch_compute(nonzero_tower)
                np.save(save_dir + '/n{}_d{}_k{}_kappa{}_beta{}.npy'.format(n, d, k, kappa, beta), efp)
        print('time', time.time() - s)

elif phase == 'merge':
    col = 0
    efp_merge = np.zeros((105540, len(graphs) * 9))

    for efp_ix, graph in enumerate(graphs):
        n, e, d, v, k, c, p, h = efpset.specs[efp_ix]
        print('efp_ix', efp_ix, 'ndk', n, d, k)
        for kappa in kappas:
            for beta in betas:
                print(beta, kappa)
                # if 'n{}_d{}_k{}_kappa{}_beta{}'.format(n, d, k, kappa, beta) == 'n4_d6_k21_kappa1_beta2':
                #     continue
                efp = np.load(save_dir + '/n{}_d{}_k{}_kappa{}_beta{}.npy'.format(n, d, k, kappa, beta))
                efp_merge[:, col] = efp
                col += 1

    efp_merge_normalized = (efp_merge - efp_merge.mean(axis=0)) / efp_merge.std(axis=0)

    with h5py.File(save_dir + '/efp_merge.h5', 'a') as f:
        f.create_dataset('efp_merge', data=efp_merge)
        f.create_dataset('efp_merge_normalized', data=efp_merge_normalized)
    print('finish saving merging')