import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import compare_data_stats as cmp
from Kidera import kidera
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 6 plots:
# 1. TCR length distribution in McPAS-TCR and TCRGP backgrounds
# 2. Difference of correlation matrices as 3d bar
# 3. Kidera avg 7th factor histogram
# 4. empty (autoencoder model)
# 5. empty (lstm model)
# 6. ROC curves (ae/lstm * internal/externals, McPAS-TCR)

w = 'McPAS-TCR_with_V'
t = 'TCRGP_with_V'
nt = 'TCRGP_negs_with_V'


def tcr_length_dist_comp(ax, data1, data2, title, normalize=False):
    with open(data1, 'r') as data:
        lens1 = {}
        for line in data:
            t = line.strip().split()[0]
            try:
                lens1[len(t)] += 1
            except KeyError:
                lens1[len(t)] = 1
    with open(data2, 'r') as data:
        lens2 = {}
        for line in data:
            t = line.strip().split()[0]
            try:
                lens2[len(t)] += 1
            except KeyError:
                lens2[len(t)] = 1
    x1 = [k for k in sorted(lens1)]
    y1 = [lens1[k] for k in sorted(lens1)]
    x2 = [k for k in sorted(lens2)]
    y2 = [lens2[k] for k in sorted(lens2)]
    if normalize:
        y1 = [y/sum(y1) for y in y1]
        y2 = [y/sum(y2) for y in y2]
    width = 0.35
    plot1 = ax.bar([x + width for x in x1], y1, width,
                    color='dodgerblue', label='McPAS')
    plot2 = ax.bar(x2, y2, width,
                    color='salmon', label='TCRGP')
    ax.set_ylabel('Number of TCRs', fontdict={'fontsize': 14})
    ax.set_title(title, fontdict={'fontsize': 16})
    ax.set_xticks(x1)
    ax.legend()


def amino_corr_map_comp(ax, data1, data2, c, title):
    amino_acids = ['start'] + [letter for letter in 'ARNDCEQGHILKMFPSTWYV'] + ['end']
    amino_to_ix = {aa: index for index, aa in enumerate(amino_acids)}
    # P[i, j] = p(j | i) stochastic transition matrix as in markov models
    with open(data1, 'r') as data:
        P1 = np.zeros([22, 22])
        for line in data:
            if c == 'tcr':
                t = line.strip().split()[0]
            if c == 'pep':
                t = line.strip().split()[1]
            if '*' in t or '*' in t:
                continue
            if '/' in t:
                continue
            # REMOVE INITIAL 'CAS' AND 'F' SUFFIX
            t = t[3:-1]
            P1[amino_to_ix['start'], amino_to_ix[t[0]]] += 1
            for i in range(len(t) - 1):
                P1[amino_to_ix[t[i]], amino_to_ix[t[i+1]]] += 1
            P1[amino_to_ix[t[-1]], amino_to_ix['end']] += 1
        P1 = P1.astype('float') / P1.sum(axis=1)[:, np.newaxis]
        print(P1)
    with open(data2, 'r') as data:
        P2 = np.zeros([22, 22])
        for line in data:
            if c == 'tcr':
                t = line.strip().split()[0]
            if c == 'pep':
                t = line.strip().split()[1]
            if '*' in t or '*' in t:
                continue
            if '/' in t:
                continue
            # REMOVE INITIAL 'CAS' AND 'F' SUFFIX
            t = t[3:-1]
            P2[amino_to_ix['start'], amino_to_ix[t[0]]] += 1
            for i in range(len(t) - 1):
                P2[amino_to_ix[t[i]], amino_to_ix[t[i+1]]] += 1
            P2[amino_to_ix[t[-1]], amino_to_ix['end']] += 1
        P2 = P2.astype('float') / P2.sum(axis=1)[:, np.newaxis]
        print(P2)
    # P = np.log(P1 / P2)
    P = P1 - P2
    mat = ax.matshow(P, cmap='bwr', vmin=-0.2, vmax=0.2)
    ax.set_xticks(range(22))
    ax.set_xticklabels(amino_acids)
    ax.set_yticks(range(22))
    ax.set_yticklabels(amino_acids)
    plt.colorbar(mat)
    ax.set_title(title, fontdict={'fontsize': 16})


def amino_corr_map_comp_3d(ax, data1, data2, c, title):
    amino_acids = ['start'] + [letter for letter in 'ARNDCEQGHILKMFPSTWYV'] + ['end']
    amino_to_ix = {aa: index for index, aa in enumerate(amino_acids)}
    # P[i, j] = p(j | i) stochastic transition matrix as in markov models
    with open(data1, 'r') as data:
        P1 = np.zeros([22, 22])
        for line in data:
            if c == 'tcr':
                t = line.strip().split()[0]
            if c == 'pep':
                t = line.strip().split()[1]
            if '*' in t or '*' in t:
                continue
            if '/' in t:
                continue
            # REMOVE INITIAL 'CAS' AND 'F' SUFFIX
            t = t[3:-1]
            P1[amino_to_ix['start'], amino_to_ix[t[0]]] += 1
            for i in range(len(t) - 1):
                P1[amino_to_ix[t[i]], amino_to_ix[t[i+1]]] += 1
            P1[amino_to_ix[t[-1]], amino_to_ix['end']] += 1
        P1 = P1.astype('float') / P1.sum(axis=1)[:, np.newaxis]
        # print(P1)
    with open(data2, 'r') as data:
        P2 = np.zeros([22, 22])
        for line in data:
            if c == 'tcr':
                t = line.strip().split()[0]
            if c == 'pep':
                t = line.strip().split()[1]
            if '*' in t or '*' in t:
                continue
            if '/' in t:
                continue
            # REMOVE INITIAL 'CAS' AND 'F' SUFFIX
            t = t[3:-1]
            P2[amino_to_ix['start'], amino_to_ix[t[0]]] += 1
            for i in range(len(t) - 1):
                P2[amino_to_ix[t[i]], amino_to_ix[t[i+1]]] += 1
            P2[amino_to_ix[t[-1]], amino_to_ix['end']] += 1
        P2 = P2.astype('float') / P2.sum(axis=1)[:, np.newaxis]
        # print(P2)
    # P = np.log(P1 / P2)
    P = P1 - P2
    _xx, _yy = np.meshgrid(range(22), range(22))
    x, y = _xx.ravel(), _yy.ravel()
    bottom = np.zeros_like(P.ravel())
    ax.bar3d(x, y, bottom, 0.3, 0.3, P.ravel(), color='dodgerblue')
    ax.set_xticks(range(22))
    ax.set_xticklabels(amino_acids)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    ax.set_yticks(range(22))
    ax.set_yticklabels(amino_acids)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    ax.set_zlim(-0.2, 0.2)
    # plt.colorbar()
    ax.set_title(title)


def kidera_hist7(ax, data1, data2):
    factor_observations1 = [[] for i in range(10)]
    with open(data1, 'r') as data:
        for line in data:
            line = line.split('\t')
            tcr = line[0]
            tcr = tcr[3:-1]
            v = kidera.score_sequence(tcr)
            v = v.values
            for i in range(len(v)):
                factor_observations1[i].append(v[i])
    factor_observations2 = [[] for i in range(10)]
    with open(data2, 'r') as data:
        for line in data:
            line = line.split('\t')
            tcr = line[0]
            tcr = tcr[3:-1]
            v = kidera.score_sequence(tcr)
            v = v.values
            for i in range(len(v)):
                factor_observations2[i].append(v[i])
    i = 7 - 1
    a = factor_observations1[i]
    b = factor_observations2[i]
    weights1 = np.ones_like(a) / float(len(a))
    weights2 = np.ones_like(b) / float(len(b))
    bins = np.linspace(-1.0, 1.0, 10)
    plot1 = ax.hist(a, weights=weights1, bins=bins,
                    color='dodgerblue', alpha=0.5, label='McPAS', width=0.05)
    plot2 = ax.hist(b, weights=weights2, bins=bins,
                    color='salmon', alpha=0.5, label='TCRGP', width=0.05)
    ax.set_title('Kidera ' + str(i+1) + ' factor normalized histogram')
    ax.legend()


def model_graph(ax, image):
    img = mpimg.imread(image)
    ax.imshow(img)
    ax.axis('off')


def plot_roc(ax, title, files, labels, colors, lns):
    for file, label, color, ln in zip(files, labels, colors, lns):
        roc = np.load(file)
        ax.plot(roc['fpr'], roc['tpr'], label=label + ', AUC=' + str(format(roc['auc'].item(), '.3f')),
                 color=color, linestyle=ln)
    plt.title(title, fontdict={'fontsize': 16})
    ax.set_xlabel('False positive rate', fontdict={'fontsize': 14})
    ax.set_ylabel('True positive rate', fontdict={'fontsize': 14})
    ax.legend()


def main():
    fig = plt.figure(1)
    ax = fig.add_subplot(231)
    # model_graph(ax, 'lstm_draft.png')
    ax.axis('off')
    ax.text(-0.1, 1.1, 'A', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    ax = fig.add_subplot(232)
    ax.axis('off')
    # model_graph(ax, 'autoencoder_draft.png')
    ax.text(-0.1, 1.1, 'B', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    ax = fig.add_subplot(233)
    plot_roc(ax, 'Models ROC curve on McPAS-TCR',
             ['ae_roc_w_gp2.npz', 'ae_roc_w2.npz', 'lstm_roc_w_gp2.npz', 'lstm_roc_w2.npz'],
             ['AE, externals', 'AE, internals', 'LSTM, externals', 'LSTM, internals'],
             ['salmon', 'dodgerblue', 'salmon', 'dodgerblue'],
             ['-', '-', '--', '--'])
    ax.text(-0.1, 1.1, 'C', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    ax = fig.add_subplot(234)
    tcr_length_dist_comp(ax, w, nt, 'Normalized TCR Length Distribution', normalize=True)
    # ax = fig.add_subplot(235, projection='3d')
    # amino_corr_map_comp_3d(ax, w, nt, 'tcr', 'Amino acids correlation maps difference')
    ax.text(-0.1, 1.1, 'D', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    ax = fig.add_subplot(235)
    amino_corr_map_comp(ax, w, nt, 'tcr', 'Amino acids correlation maps difference')
    ax.text(-0.1, 1.06, 'E', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    ax = fig.add_subplot(236)
    ax.axis('off')
    # kidera_hist7(ax, w, nt)
    ax.text(-0.1, 1.1, 'F', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
    plt.show()
    plt.tight_layout()
    plt.show()
    pass


if __name__ == '__main__':
    main()
