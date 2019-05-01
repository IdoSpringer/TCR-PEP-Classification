import pair_sampling.pairs_data.stats as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import torch
import pickle
from ae_pep_cd_test_eval import *
from scipy import stats

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

    ax.set_ylabel('Log TCRs per peptide', fontdict={'fontsize': 14})
    ax.set_xlabel('Peptide index', fontdict={'fontsize': 14})
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
    ax.set_xlabel('Number of TCR-peptide pairs / 1000', fontdict={'fontsize': 11}, labelpad=1)
    ax.set_ylabel('Mean AUC score', fontdict={'fontsize': 14})
    ax.set_title(title, fontdict={'fontsize': 16})
    ax.legend()


def plot_roc(ax, title, files, labels, colors, lns):
    for file, label, color, ln in zip(files, labels, colors, lns):
        roc = np.load(file)
        ax.plot(roc['fpr'], roc['tpr'], label=label + ', AUC=' + str(format(roc['auc'].item(), '.3f')),
                 color=color, linestyle=ln)
    plt.title(title, fontdict={'fontsize': 16})
    ax.set_xlabel('False positive rate', fontdict={'fontsize': 14})
    ax.set_ylabel('True positive rate', fontdict={'fontsize': 14})
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
        if mkey == 'lstm' and dkey == 's' and mis > 27:
            continue
        state = name[-3]
        if state == 'test' or state == 'test2':
            auc = max_auc(dir + '/' + filename)
            aucs.append((mkeys[mkey], dkeys[dkey], iteration, mis, auc))
    max_index0 = max(t[3] for t in aucs if t[0] == 0)
    max_index10 = max(t[3] for t in aucs if t[0] == 1 and t[1] == 0)
    # max_index11 = max(t[3] for t in aucs if t[0] == 1 and t[1] == 1)
    max_index11 = 27
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
    labels = ['MsPAS, AE model', 'VDJdb, AE model', 'MsPAS, LSTM model', 'VDJdb, LSTM model']
    colors = ['dodgerblue', 'springgreen', 'dodgerblue', 'springgreen']
    styles = ['-', '-', '--', '--']
    for auc_mean, auc_std, label, color, style in zip(auc_means, auc_stds, labels, colors, styles):
        ax.errorbar(range(1, len(auc_mean) + 1), auc_mean, yerr=auc_std, label=label,
                    color=color, linestyle=style)
    ax.legend(loc=4, prop={'size': 8})
    ax.set_xlabel('Missing amino acid index', fontdict={'fontsize': 11}, labelpad=1)
    ax.set_ylabel('Best AUC score', fontdict={'fontsize': 14})
    ax.set_title(title, fontdict={'fontsize': 16})
    pass


def auc_per_pep_num_tcrs(ax, device):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}
    args = {}
    args['ae_file'] = 'pad_full_data_autoencoder_model1.pt'
    params = {}
    params['lr'] = 1e-3
    params['wd'] = 1e-5
    params['epochs'] = 200
    params['emb_dim'] = 10
    params['enc_dim'] = 30
    params['dropout'] = 0.1
    params['train_ae'] = False
    # Load autoencoder params
    checkpoint = torch.load(args['ae_file'])
    params['max_len'] = checkpoint['max_len']
    params['batch_size'] = checkpoint['batch_size']
    batch_size = params['batch_size']

    directory = 'test_and_models_with_cd/'
    auc_mat = np.zeros((10, 8))
    for iteration in range(10):
        # load test
        test_file = directory + 'ae_test_w_' + str(iteration)
        model_file = directory + 'ae_model_w_' + str(iteration)
        device = device
        with open(test_file, 'rb') as fp:
            test = pickle.load(fp)
        # test
        test_tcrs, test_peps, test_signs = get_lists_from_pairs(test, params['max_len'])
        test_batches = get_batches(test_tcrs, test_peps, test_signs, tcr_atox, pep_atox, params['batch_size'],
                                   params['max_len'])
        # load model
        model = AutoencoderLSTMClassifier(params['emb_dim'], device, params['max_len'], 21, params['enc_dim'],
                                          params['batch_size'], args['ae_file'], params['train_ae'])
        trained_model = torch.load(model_file)
        model.load_state_dict(trained_model['model_state_dict'])
        model.eval()
        model = model.to(device)

        peps_pos_probs = {}
        for i in range(len(test_batches)):
            batch = test_batches[i]
            batch_data = test[i * batch_size: (i + 1) * batch_size]
            tcrs, padded_peps, pep_lens, batch_signs = batch
            # Move to GPU
            tcrs = torch.tensor(tcrs).to(device)
            padded_peps = padded_peps.to(device)
            pep_lens = pep_lens.to(device)
            probs = model(tcrs, padded_peps, pep_lens)
            peps = [data[1] for data in batch_data]
            # cd = [data[2] for data in batch_data]
            for pep, prob, sign in zip(peps, probs, batch_signs):
                try:
                    peps_pos_probs[pep].append((prob.item(), sign))
                except KeyError:
                    peps_pos_probs[pep] = [(prob.item(), sign)]
        bins = {}
        for pep in peps_pos_probs:
            num_examples = len(peps_pos_probs[pep])
            bin = int(np.floor(np.log2(num_examples)))
            try:
                bins[bin].extend(peps_pos_probs[pep])
            except KeyError:
                bins[bin] = peps_pos_probs[pep]
        for bin in bins:
            pass
            # print(bin, len(bins[bin]))
        bin_aucs = {}
        for bin in bins:
            try:
                auc = roc_auc_score([p[1] for p in bins[bin]], [p[0] for p in bins[bin]])
                bin_aucs[bin] = auc
                # print(bin, auc)
            except ValueError:
                # print(bin, [p[1] for p in bins[bin]])
                pass
        bin_aucs = sorted(bin_aucs.items())
        # print(bin_aucs)
        auc_mat[iteration] = np.array([t[1] for t in bin_aucs])
        pass
    # print(auc_mat)
    means = np.mean(auc_mat, axis=0)
    std = stats.sem(auc_mat, axis=0)
    # print(means, std)
    ax.errorbar([j[0] for j in bin_aucs], means, yerr=std, color='dodgerblue')
    ax.set_xticks([j[0] for j in bin_aucs])
    ax.set_xticklabels([2 ** j[0] for j in bin_aucs])
    ax.set_xlabel('Number of peptide TCRs bins', fontdict={'fontsize': 14})
    ax.set_ylabel('Averaged AUC score', fontdict={'fontsize': 14})
    ax.set_title('AUC per number of TCRs per peptide', fontdict={'fontsize': 16})
    pass


def main():
    fig = plt.figure(2)
    ax = fig.add_subplot(231)
    subsamples_auc(ax, 'w', 's', 'AUC per number of pairs')
    ax.text(-0.1, 1.1, 'A', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    ax = fig.add_subplot(232)
    plot_roc(ax, 'Models ROC curve on cancer dataset',
             ['ae_roc_exc_gp2.npz', 'ae_roc_exc2.npz', 'lstm_roc_exc_gp2.npz', 'lstm_roc_exc2.npz'],
             ['AE, externals', 'AE, internals', 'LSTM, externals', 'LSTM, internals'],
             ['salmon', 'orchid', 'salmon', 'orchid'],
             ['-', '-', '--', '--'])
    ax.text(-0.1, 1.1, 'B', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    ax = fig.add_subplot(233)
    position_auc(ax, 'AUC per missing amino acids')
    ax.text(-0.1, 1.1, 'C', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    ax = fig.add_subplot(234)
    tcr_per_pep_dist(ax, w, s, 'Number of TCRs pep peptide')
    ax.text(-0.1, 1.1, 'D', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    ax = fig.add_subplot(235)
    ax.axis('off')
    ax.text(-0.1, 1.1, 'E', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    ax = fig.add_subplot(236)
    auc_per_pep_num_tcrs(ax, 'cuda:0')
    ax.text(-0.1, 1.1, 'F', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    plt.tight_layout()
    plt.show()
    pass


if __name__ == '__main__':
    main()
