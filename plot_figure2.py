import pair_sampling.pairs_data.stats as st
import numpy as np
import matplotlib.pyplot as plt
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

    ax.bar(range(len(tcr_nums1)), np.log(np.array(tcr_nums1)),
           color='orchid', alpha=0.5, label='McPAS')
    ax.bar(range(len(tcr_nums2)), np.log(np.array(tcr_nums2)),
           color='springgreen', alpha=0.5, label='VDJdb')

    ax.set_ylabel('log TCRs per peptide')
    ax.set_xlabel('peptide index')
    ax.set_title(title)
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
            if iteration > 5:
                continue
            auc = max_auc(dir + '/' + filename)
            aucs1.append((iteration, sub_index, auc))
    max_index1 = max(t[1] for t in aucs1)
    max_iter1 = max(t[0] for t in aucs1)
    auc_matrix1 = np.zeros((max_iter1 + 1, max_index1))
    for auc in aucs1:
        auc_matrix1[auc[0], auc[1] - 1] = auc[2]
    means1 = np.mean(auc_matrix1, axis=0)
    stds1 = np.std(auc_matrix1, axis=0)

    aucs2 = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith('ae_' + key2 + '_test_sub'):
            sub_index = int(filename.split('_')[-1])
            iteration = int(filename.split('_')[-2])
            if iteration > 0:
                continue
            auc = max_auc(dir + '/' + filename)
            aucs2.append((iteration, sub_index, auc))
    max_index2 = max(t[1] for t in aucs2)
    max_iter2 = max(t[0] for t in aucs2)
    auc_matrix2 = np.zeros((max_iter2 + 1, max_index2))
    for auc in aucs2:
        auc_matrix2[auc[0], auc[1] - 1] = auc[2]
    means2 = np.mean(auc_matrix2, axis=0)
    stds2 = np.std(auc_matrix2, axis=0)

    ax.errorbar(range(max_index1), means1, yerr=stds1, color='orchid', label='McPAS')
    ax.errorbar(range(max_index2), means2, yerr=stds2, color='springgreen', label='VDJdb')
    ax.set_xlabel('Number of TCR-peptide pairs / 1000')
    ax.set_ylabel('Mean AUC score')
    ax.set_title(title)
    ax.legend()


def plot_roc(ax, title, files, labels, colors):
    for file, label, color in zip(files, labels, colors):
        roc = np.load(file)
        ax.plot(roc['fpr'], roc['tpr'], label=label + ', AUC=' + str(format(roc['auc'].item(), '.3f')),
                 color=color)
    plt.title(title)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.legend()


def main():
    fig = plt.figure(2)
    ax = fig.add_subplot(231)
    tcr_per_pep_dist(ax, w, s, 'Number of TCRs pep peptide')
    ax = fig.add_subplot(232)
    subsamples_auc(ax, 'w', 's', 'AUC per number of pairs')
    ax = fig.add_subplot(233)
    plot_roc(ax, 'Models ROC curve on cancer dataset',
             ['ae_roc_exc_gp2.npz', 'ae_roc_exc2.npz', 'lstm_roc_exc_gp2.npz', 'lstm_roc_exc2.npz'],
             ['ae, externals', 'ae, internals', 'lstm, externals', 'lstm, internals'],
             ['salmon', 'dodgerblue', 'tomato', 'orchid'])

    ax = fig.add_subplot(234)
    ax.axis
    ax = fig.add_subplot(235)
    ax = fig.add_subplot(236)
    # plt.tight_layout()
    plt.show()
    pass


if __name__ == '__main__':
    main()
