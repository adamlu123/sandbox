import numpy as np
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
import glob

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--subset',
    type=str,
    default='res1')
args = parser.parse_args()

def get_files():
    files = glob.glob(data_dir)
    name_dict = {N:[] for N in [1,2,3,4,6,8]}
    for file in files:
        if 'aug_uu_homog' in file:
            name_dict[1].append(file)
        elif 'aug_zww_homog' in file:
            name_dict[2].append(file)
        elif 'aug_ztt_homog' in file:
            name_dict[3].append(file)
        elif 'aug_ghha_a1000' in file:
            name_dict[4].append(file)
        elif 'aug_gtta_u_homog' in file:
            name_dict[6].append(file)
        elif 'aug_atthu_a1000' in file:
            name_dict[8].append(file)
        else:
            raise ValueError('Unknown File!')
    return name_dict


def process_HL(lines, target):
    """ input contains 3 lines, output a 1d vector
    """
    HL = np.empty(20)
    nums = lines[1].split()[4::2] + lines[2].split()[4::2] + lines[0].split()[2::2] + [target]
    return list(map(float, nums))


def process_tower(towerslines_by_jet):
    """ input a list of strings, extract the pT, eta and phi information
    """
    towerslist_by_jet = []
    for line in towerslines_by_jet:
        towerslist_by_jet.append(list(map(float, line.split()[-3:])))
    return towerslist_by_jet


def parser(file, N, mass_range):
    HL = []
    Tower = []
    towerslines_by_jet = []
    cnt = 0
    matched_cnt = 0
    jet_cnt = 0
    match = True
    for i, line in enumerate(file):
        if line[:3] == 'Jet':
            #             if len(HL) > 10:
            #                 break
            jet_cnt += 1
            ## process tower from last jet
            if towerslines_by_jet:
                Tower.append(process_tower(towerslines_by_jet))

                ### process HL for new jet
            cnt += 1
            processed = process_HL(file[i:i + 3], N)
            if processed[-3] + processed[-2] == N and (processed[-5] > mass_range[0] and processed[-5] < mass_range[1]):
                matched_cnt += 1
                HL.append(processed)
                match = True
            else:
                match = False

            ### clear cache for storing new towers
            towerslines_by_jet = []

        ### append tower lines if it is a match
        elif line[:5] == 'Tower' and match == True:
            towerslines_by_jet.append(line)

    ### process the final jets
    if towerslines_by_jet:
        Tower.append(process_tower(towerslines_by_jet))
    print('N={} total num of jets: {}, matched jets:{}'.format(N, cnt, matched_cnt))
    return HL, Tower


def parser_HL(file, N, mass_range):
    HL = []
    matched_cnt = 0
    cnt = 0
    for i, line in enumerate(file):
        if line[:3] == 'Jet':
            cnt += 1
            processed = process_HL(file[i:i + 3], N)
            if processed[-3] + processed[-2] == N and (processed[-5] > mass_range[0] and processed[-5] < mass_range[1]):
                matched_cnt += 1
                HL.append(processed)
    print('N={} total num of jets: {}, matched jets:{}'.format(N, cnt, matched_cnt))
    return HL


def save_h5(seed=123, HL_only=True, mass_range=[300, 700]):
    if not HL_only and not mass_range: raise ValueError('Missing mass_range!')
    np.random.seed(seed)

    parsed_HL = []
    parsed_Tower = []
    target = []
    ### loop over different txt files
    for N in [1, 2, 3, 4, 6, 8]:
        for i in range(len(name_dict[N])):
            data_dir = name_dict[N][i]
            file = []
            with open(data_dir, 'r') as f:
                for idx, line in enumerate(f):
                    file.append(line)
            if HL_only:
                HL = parser_HL(file, N, mass_range)
            else:
                HL, Tower = parser(file, N, mass_range)
                if len(HL) != len(Tower):
                    print(len(HL), len(Tower))
                    raise ValueError('len of HL must equal to len of Tower!')
                parsed_Tower = parsed_Tower + Tower
            print(np.array(HL)[:, -4].max())
            parsed_HL = parsed_HL + HL
            target = target + [N] * len(HL)

    idx = np.random.permutation(len(target))
    if HL_only:
        return np.array(parsed_HL)[idx], np.array(target)[idx]
    else:
        return np.array(parsed_HL)[idx], np.array(parsed_Tower)[idx], np.array(target)[idx]



############ start main
subset = args.subset
data_dir= "/baldig/physicsprojects2/N_tagger/{}/*.txt".format(subset)
files = glob.glob(data_dir)
print(len(files))
name_dict = get_files()

parsed_HL, parsed_Tower, target = save_h5(HL_only=False, mass_range=[300, 700])

## get id of the selected images
mass_range=[300, 700]
mass = parsed_HL[:, -5]
mass_list = [mass[np.where(parsed_HL[:, -1]==i)[0]] for i in [1,2,3,4,6,8]]
fig, ax = plt.subplots(2,1, figsize=(10, 10))
alpha=0.3
col = ['blue', 'orange', 'green', 'purple', 'black', 'red']
prong = [1,2,3,4,6,8]
mass_bins = np.linspace(mass_range[0], mass_range[1], 11)
cnt_list = []
for i in range(0,len(mass_list)):
    sns.distplot(mass_list[i], label='N={}'.format(prong[i]), color=col[i], hist=False, ax=ax[0])
    cnt = ax[1].hist(mass_list[i], bins=mass_bins, label='N={}'.format(prong[i]), alpha=alpha, color=col[i])
    cnt_list.append(cnt[0])
cnt_list = np.array(cnt_list)

num_per_bin = int(cnt_list.min())
bin_idx = (mass - mass_range[0]) // ((mass_range[1] - mass_range[0])/10) # mass // 40
b_and_u_subset_idx = [[np.where((bin_idx==i) & (parsed_HL[:, -1]==N))[0][:num_per_bin].tolist() for i in range(10)] for N in [1,2,3,4,6,8]]
b_and_u_subset_idx = np.asarray(b_and_u_subset_idx)
print('samples per bin:', num_per_bin, b_and_u_subset_idx.shape)

# transformed target
maps = {
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '6': 4,
    '8': 5
}
trasformed_target = []
for i in target:
    key = str(i)
    assert key in maps
    trasformed_target.append(maps[key])
print([len(np.where(trasformed_target==N)[0]) for N in np.arange(6)])
trasformed_target = np.array(trasformed_target)

padded_tower = np.zeros((b_and_u_subset_idx.reshape(-1).shape[0], 300, 3))
for i, tower in enumerate(parsed_Tower[b_and_u_subset_idx.reshape(-1)].tolist()):
    padded_tower[i, :min(len(tower), 300), :] = np.array(tower)[:min(len(tower), 300), :]

# save
print('start to save')
with h5py.File('/baldig/physicsprojects2/N_tagger/merged/{}_mass300_700_b_u_parsedTower.h5'.format(subset), 'a') as f:
    f.create_dataset('HL', data=parsed_HL[b_and_u_subset_idx.reshape(-1)])
    f.create_dataset('target', data=trasformed_target[b_and_u_subset_idx.reshape(-1)])
    f.create_dataset('parsed_Tower', data=padded_tower)
print('finish saving!', subset)