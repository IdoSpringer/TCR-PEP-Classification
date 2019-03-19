import matplotlib.pyplot as plt
import numpy as np
from Kidera import kidera

w = 'McPAS-TCR_with_V'
t = 'TCRGP_with_V'
nt = 'TCRGP_negs_with_V'


def tcr_length_dist_comp(data1, data2, title, normalize=False):
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
    fig, ax = plt.subplots()
    width = 0.35
    plot1 = ax.bar([x + width for x in x1], y1, width,
                    color='SkyBlue', label='McPAS (Weizmann) data')
    plot2 = ax.bar(x2, y2, width,
                    color='IndianRed', label='TCRGP paper data')
    ax.set_ylabel('Number of TCRs')
    ax.set_title(title)
    ax.set_xticks(x1)
    ax.legend()
    plt.show()

# tcr_length_dist_comp(w, t, 'TCR Length Distribution')
# tcr_length_dist_comp(w, t, 'Normalized TCR Length Distribution', normalize=True)
# tcr_length_dist_comp(w, nt, 'Normalized TCR Length Distribution' ,normalize=True)


def amino_corr_map(datafile, c, title):
    with open(datafile, 'r') as data:
        amino_acids = ['start'] + [letter for letter in 'ARNDCEQGHILKMFPSTWYV'] + ['end']
        amino_to_ix = {aa: index for index, aa in enumerate(amino_acids)}
        # P[i, j] = p(j | i) stochastic transition matrix as in markov models
        P = np.zeros([22, 22])
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
            P[amino_to_ix['start'], amino_to_ix[t[0]]] += 1
            for i in range(len(t) - 1):
                P[amino_to_ix[t[i]], amino_to_ix[t[i+1]]] += 1
            P[amino_to_ix[t[-1]], amino_to_ix['end']] += 1
        P = P.astype('float') / P.sum(axis=1)[:, np.newaxis]
        # for i in range(22)
        print(P)
        plt.matshow(P)
        plt.xticks(range(22), amino_acids)
        plt.yticks(range(22), amino_acids)
        plt.title(title)
        plt.show()
    pass


def amino_corr_map_comp(data1, data2, c, title):
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
    plt.matshow(P)
    plt.xticks(range(22), amino_acids)
    plt.yticks(range(22), amino_acids)
    plt.colorbar()
    plt.title(title)
    plt.show()


# amino_corr_map(w, 'tcr', 'Amino acids correlation map in McPAS data')
# amino_corr_map(t, 'tcr', 'Amino acids correlation map in TCRGP data')
# amino_corr_map_comp(w, t, 'tcr', 'Amino acids correlation maps, P1 - P2')

# amino_corr_map(w, 'tcr', 'Amino acids correlation map in McPAS data')
# amino_corr_map(nt, 'tcr', 'Amino acids correlation map in TCRGP data')
# amino_corr_map_comp(w, nt, 'tcr', 'Amino acids correlation maps, P1 - P2')


def amino_acids_distribution(data1, data2, title, normalize=False):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    with open(data2, 'r') as data:
        amino_count2 = {aa: 0 for aa in amino_acids}
        for line in data:
            t = line.strip().split()[0]
            if '*' in t or '*' in t:
                continue
            if '/' in t:
                continue
            for aa in t:
                amino_count2[aa] += 1
    with open(data1, 'r') as data:
        amino_count1 = {aa: 0 for aa in amino_acids}
        for line in data:
            t = line.strip().split()[0]
            if '*' in t or '*' in t:
                continue
            if '/' in t:
                continue
            for aa in t:
                amino_count1[aa] += 1
    x = np.arange(len(amino_acids))
    y1 = [amino_count1[k] for k in amino_acids]
    y2 = [amino_count2[k] for k in amino_acids]
    if normalize:
        y1 = [y/sum(y1) for y in y1]
        y2 = [y/sum(y2) for y in y2]
    fig, ax = plt.subplots()
    width = 0.35
    plot1 = ax.bar([key + width for key in x], y1, width,
                   color='SkyBlue', label='McPAS (Weizmann) data')
    plot2 = ax.bar(x, y2, width,
                   color='IndianRed', label='TCRGP paper data')
    ax.set_ylabel('Number of amino acids')
    ax.set_title(title)
    plt.xticks(x, amino_acids)
    ax.legend()
    plt.show()


# amino_acids_distribution(w, t, 'Animo Acids Distribution')
# amino_acids_distribution(w, t, 'Normalized Animo Acids Distribution', normalize=True)
# amino_acids_distribution(w, nt, 'Normalized Animo Acids Distribution', normalize=True)


def v_gene_dist(data1, data2, title, normalize=False):
    with open(data2, 'r') as data:
        v_count2 = {}
        for line in data:
            v = line.strip().split()[1]
            #if '*' in t or '*' in t:
            #    continue
            #if '/' in t:
            #    continue
            try:
                v_count2[v] += 1
            except KeyError:
                v_count2[v] = 1
    '''
    with open(data1, 'r') as data:
        amino_count1 = {aa: 0 for aa in amino_acids}
        for line in data:
            t = line.strip().split()[0]
            if '*' in t or '*' in t:
                continue
            if '/' in t:
                continue
            for aa in t:
                amino_count1[aa] += 1
    '''
    print(v_count2)
    x = np.arange(len(v_count2))
    y1 = [amino_count1[k] for k in amino_acids]
    y2 = [amino_count2[k] for k in amino_acids]
    if normalize:
        y1 = [y/sum(y1) for y in y1]
        y2 = [y/sum(y2) for y in y2]
    fig, ax = plt.subplots()
    width = 0.35
    plot1 = ax.bar([key + width for key in x], y1, width,
                   color='SkyBlue', label='McPAS (Weizmann) data')
    plot2 = ax.bar(x, y2, width,
                   color='IndianRed', label='TCRGP paper data')
    ax.set_ylabel('Number of amino acids')
    ax.set_title(title)
    plt.xticks(x, amino_acids)
    ax.legend()
    plt.show()
    pass


def avg_kidera_score(datafile):
    with open(datafile, 'r') as data:
        index = 0
        avg = np.zeros((10))
        for line in data:
            line = line.split('\t')
            tcr = line[0]
            tcr = tcr[3:-1]
            v = kidera.score_sequence(tcr)
            v = v.values
            avg += v
            index += 1
    avg /= index
    return avg


def plot_kidera():
    w_avg = avg_kidera_score(w)
    nt_avg = avg_kidera_score(nt)
    print('weizmann avg:', w_avg)
    print('TCRGP avg:', nt_avg)
    x = np.arange(10)
    fig, ax = plt.subplots()
    width = 0.35
    plot1 = ax.bar([t + width for t in x], w_avg, width,
                   color='SkyBlue', label='McPAS (Weizmann) data')
    plot2 = ax.bar(x, nt_avg, width,
                   color='IndianRed', label='TCRGP paper data')
    ax.set_ylabel('Average Kidera score')
    ax.set_title('Average Kidera Score')
    ax.set_xticks(x)
    ax.set_xlabel('Kidera factor')
    ax.legend()
    plt.show()


# plot_kidera()


def kidera_hist(data1, data2):
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
    for i in range(len(factor_observations1)):
        fig, ax = plt.subplots()
        a = factor_observations1[i]
        b = factor_observations2[i]
        weights1 = np.ones_like(a) / float(len(a))
        weights2 = np.ones_like(b) / float(len(b))
        plot1 = ax.hist(a, weights=weights1,
                       color='SkyBlue', alpha=0.5, label='McPAS (Weizmann) data')
        plot2 = ax.hist(b, weights=weights2,
                       color='IndianRed', alpha=0.5, label='TCRGP paper data')
        #ax.set_ylabel('Number of TCRs')
        ax.set_title('Kidera ' + str(i+1) + ' factor normalized histogram')
        #ax.set_xticks(x1)
        ax.legend()
        # plt.show()
        plt.savefig('stats_compare_plots/kidera_factors/kidera_' + str(i+1))


kidera_hist(w, nt)
