import pair_sampling.pairs_data.stats as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

w = 'pair_sampling/pairs_data/weizmann_pairs.txt'
s = 'pair_sampling/pairs_data/shugay_pairs.txt'


def tcr_per_pep_dist(ax, data1, data2, title):
    pep_tcr1 = {}
    with open(data1, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            tcr = line[0]
            pep = line[1]
            try:
                pep_tcr1[pep] += 1
            except KeyError:
                pep_tcr1[pep] = 1
    tcr_nums1 = sorted([pep_tcr1[pep] for pep in pep_tcr1], reverse=True)
    pep_tcr2 = {}
    with open(data2, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            tcr = line[0]
            pep = line[1]
            try:
                pep_tcr2[pep] += 1
            except KeyError:
                pep_tcr2[pep] = 1
    tcr_nums2 = sorted([pep_tcr2[pep] for pep in pep_tcr2], reverse=True)

    ax.plot(range(len(tcr_nums1)), np.log(np.array(tcr_nums1)),
           color='orchid', label='McPAS')
    ax.plot(range(len(tcr_nums2)), np.log(np.array(tcr_nums2)),
           color='springgreen', label='VDJdb')

    ax.set_ylabel('log TCRs per peptide', fontdict={'fontsize': 12})
    ax.set_xlabel('peptide index', fontdict={'fontsize': 12})
    ax.set_title(title, fontdict={'fontsize': 16})
    ax.legend()
    pass


def max_auc(auc_file):
    with open(auc_file, 'r') as file:
        aucs = []
        for line in file:
            aucs.append(float(line.strip()))
        max_auc = max(aucs)
    return max_auc


def subsamples_auc(ax, key1, key2, title):
    dir = 'subsamples_auc'
    directory = os.fsencode(dir)
    aucs1 = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith('ae_' + key1 + '_test_sub'):
            sub_index = int(filename.split('_')[-1])
            iteration = int(filename.split('_')[-2])
            auc = max_auc(dir + '/' + filename)
            aucs1.append((iteration, sub_index, auc))
    max_index1 = max(t[1] for t in aucs1)
    max_iter1 = max(t[0] for t in aucs1)
    auc_matrix1 = np.zeros((max_iter1 + 1, max_index1))
    for auc in aucs1:
        auc_matrix1[auc[0], auc[1] - 1] = auc[2]
    means1 = np.mean(auc_matrix1, axis=0)
    stds1 = stats.sem(auc_matrix1, axis=0)

    aucs2 = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith('ae_' + key2 + '_test_sub'):
            sub_index = int(filename.split('_')[-1])
            iteration = int(filename.split('_')[-2])
            auc = max_auc(dir + '/' + filename)
            aucs2.append((iteration, sub_index, auc))
    max_index2 = max(t[1] for t in aucs2)
    max_iter2 = max(t[0] for t in aucs2)
    auc_matrix2 = np.zeros((max_iter2 + 1, max_index2))
    for auc in aucs2:
        auc_matrix2[auc[0], auc[1] - 1] = auc[2]
    means2 = np.mean(auc_matrix2, axis=0)
    stds2 = stats.sem(auc_matrix2, axis=0)

    ax.errorbar(range(max_index1)[:-1], means1[:-1], yerr=stds1[:-1], color='dodgerblue', label='McPAS')
    ax.errorbar(range(max_index2), means2, yerr=stds2, color='springgreen', label='VDJdb')
    ax.set_xlabel('Number of TCR-peptide pairs / 1000', fontdict={'fontsize': 12})
    ax.set_ylabel('Mean AUC score', fontdict={'fontsize': 12})
    ax.set_title(title, fontdict={'fontsize': 16})
    ax.legend()


def plot_roc(ax, title, files, labels, colors, lns):
    for file, label, color, ln in zip(files, labels, colors, lns):
        roc = np.load(file)
        ax.plot(roc['fpr'], roc['tpr'], label=label + ', AUC=' + str(format(roc['auc'].item(), '.3f')),
                 color=color, linestyle=ln)
    plt.title(title, fontdict={'fontsize': 16})
    ax.set_xlabel('False positive rate', fontdict={'fontsize': 12})
    ax.set_ylabel('True positive rate', fontdict={'fontsize': 12})
    ax.legend()


def position_auc(ax, title):
    dir = 'mis_pos_auc'
    mkeys = {'ae': 0, 'lstm': 1}
    dkeys = {'w': 0, 's': 1}
    directory = os.fsencode(dir)
    aucs = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        name = filename.split('_')
        if not len(name) is 6:
            continue
        mkey = name[0]
        dkey = name[1]
        mis = int(name[-1])
        iteration = int(name[-2])

        if iteration > 0 and mkey == 'lstm' and dkey == 's':
            continue
        state = name[-3]
        if state == 'test':
            auc = max_auc(dir + '/' + filename)
            aucs.append((mkeys[mkey], dkeys[dkey], iteration, mis, auc))
    max_index0 = max(t[3] for t in aucs if t[0] == 0)
    max_index10 = max(t[3] for t in aucs if t[0] == 1 and t[1] == 0)
    max_index11 = max(t[3] for t in aucs if t[0] == 1 and t[1] == 1)
    max_index = max(max_index0, max_index10, max_index11)
    max_iter0 = max(t[2] for t in aucs if t[0] == 0)
    max_iter10 = max(t[2] for t in aucs if t[0] == 1 and t[1] == 0)
    max_iter11 = max(t[2] for t in aucs if t[0] == 1 and t[1] == 1)
    max_iter = max(max_iter0, max_iter10, max_iter11)
    auc_tensor = np.zeros((2, 2, max_iter + 1, max_index + 1))
    for auc in aucs:
        auc_tensor[auc[0], auc[1], auc[2], auc[3]] = auc[4]
    auc_tensor0 = auc_tensor[0, :, :max_iter0 + 1, :max_index0 + 1]
    # print('auc tensor 0')
    # print(auc_tensor0)
    auc_tensor10 = auc_tensor[1, 0, :max_iter10 + 1, :max_index10 + 1]
    # print('auc tensor 10')
    # print(auc_tensor10)
    auc_tensor11 = auc_tensor[1, 1, :max_iter11 + 1, :max_index11 + 1]
    # print('auc tensor 11')
    # print(auc_tensor11)
    means0 = np.mean(auc_tensor0, axis=1)
    std0 = stats.sem(auc_tensor0, axis=1)
    means10 = np.mean(auc_tensor10, axis=0)
    std10 = stats.sem(auc_tensor10, axis=0)
    means11 = np.mean(auc_tensor11, axis=0)
    std11 = stats.sem(auc_tensor11, axis=0)
    auc_means = [means0[0], means0[1], means10, means11]
    # print(auc_means)
    auc_stds = [std0[0], std0[1], std10, std11]
    labels = ['MsPAS, ae model', 'VDJdb, ae model', 'MsPAS, lstm model', 'VDJdb, lstm model']
    for auc_mean, auc_std, label in zip(auc_means, auc_stds, labels):
        ax.errorbar(range(1, len(auc_mean) + 1), auc_mean, yerr=auc_std, label=label)
    ax.legend()
    ax.set_xlabel('Missing index')
    ax.set_ylabel('best AUC score')
    ax.set_title(title)
    pass


def main():
    fig = plt.figure(2)
    ax = fig.add_subplot(224)
    tcr_per_pep_dist(ax, w, s, 'Number of TCRs pep peptide')
    ax = fig.add_subplot(221)
    subsamples_auc(ax, 'w', 's', 'AUC per number of pairs')
    ax = fig.add_subplot(222)
    plot_roc(ax, 'Models ROC curve on cancer dataset',
             ['ae_roc_exc_gp2.npz', 'ae_roc_exc2.npz', 'lstm_roc_exc_gp2.npz', 'lstm_roc_exc2.npz'],
             ['AE, externals', 'AE, internals', 'LSTM, externals', 'LSTM, internals'],
             ['salmon', 'dodgerblue', 'salmon', 'dodgerblue'],
             ['-', '-', '--', '--'])
    ax = fig.add_subplot(223)
    position_auc(ax, 'AUC per missing amino acids')
    # plt.tight_layout()
    plt.show()
    pass


if __name__ == '__main__':
    main()
