import matplotlib.pyplot as plt
import numpy as np

w = 'McPAS-TCR_with_V'
t = 'TCRGP_with_V'


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


w = 'McPAS-TCR_with_V'
t = 'TCRGP_with_V'
# amino_corr_map(w, 'tcr', 'Amino acids correlation map in McPAS data')
# amino_corr_map(t, 'tcr', 'Amino acids correlation map in TCRGP data')


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
amino_acids_distribution(w, t, 'Normalized Animo Acids Distribution', normalize=True)


def v_gene_dist(data):
    pass
