import matplotlib.pyplot as plt
import numpy as np


def num_of_peps(pair_file):
    peps = set()
    l = 0
    with open(pair_file, 'r') as file:
        for line in file:
            l += 1
            pep = line.strip().split('\t')[-1]
            peps.add(pep)
    num_peps = len(peps)
    print(num_peps, l)
    return num_peps / l


def num_of_tcrs(pair_file):
    tcrs = set()
    l = 0
    with open(pair_file, 'r') as file:
        for line in file:
            l += 1
            tcr = line.strip().split('\t')[0]
            tcrs.add(tcr)
    num_tcrs = len(tcrs)
    print(num_tcrs, l)
    return num_tcrs / l


def tcr_per_pep(pair_file):
    pep_tcr = {}
    with open(pair_file, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            tcr = line[0]
            pep = line[1]
            try:
                pep_tcr[pep].append(tcr)
            except KeyError:
                pep_tcr[pep] = [tcr]
    count = 0
    for pep in pep_tcr:
        if len(pep_tcr[pep]) >= 10:
            count += 1
    return count


def length_dist(datafile, c,  title):
    with open(datafile, 'r') as data:
        lens = {}
        for line in data:
            if c == 'tcr':
                t = line.strip().split()[0]
            if c == 'pep':
                t = line.strip().split()[1]
            try:
                lens[len(t)] += 1
            except KeyError:
                lens[len(t)] = 1
        # lens = sorted(lens)
        print(lens)
        m1 = min(key for key in lens)
        m2 = max(key for key in lens)
        l = [0] * len(range(m1, m2+1))
        for key in sorted(lens):
            l[key - m1] = lens[key]
        plt.bar(range(m1, m2 + 1), l)
        plt.xticks(list(range(m1, m2+1)))
        plt.title(title)
        plt.show()


def amino_acids_distribution(datafile, c,  title):
    with open(datafile, 'r') as data:
        amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        amino_count = {aa: 0 for aa in amino_acids}
        for line in data:
            if c == 'tcr':
                t = line.strip().split()[0]
            if c == 'pep':
                t = line.strip().split()[1]
            if '*' in t or '*' in t:
                continue
            if '/' in t:
                continue
            for aa in t:
                amino_count[aa] += 1
        print(amino_count)
        x = range(len(amino_acids))
        l = [amino_count[aa] for aa in amino_count]
        plt.bar(x, l)
        plt.xticks(x, amino_acids)
        plt.title(title)
        plt.show()
    pass


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
            P[amino_to_ix['start'], amino_to_ix[t[0]]] += 1
            for i in range(len(t) - 1):
                P[amino_to_ix[t[i]], amino_to_ix[t[i+1]]] += 1
            P[amino_to_ix[t[-1]], amino_to_ix['end']] += 1
        P = P.astype('float') / P.sum(axis=1)[:, np.newaxis]
        # for i in range(22)
        print(P)
        #plot_stoc_matrix(P, [], normalize=True)
        plt.matshow(P)
        plt.xticks(range(22), amino_acids)
        plt.yticks(range(22), amino_acids)
        plt.title(title)
        plt.show()
    pass


def tcr_per_pep_dist(datafile, title):
    pep_tcr = {}
    with open(datafile, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            tcr = line[0]
            pep = line[1]
            try:
                pep_tcr[pep] += 1
            except KeyError:
                pep_tcr[pep] = 1
    tcr_nums = sorted([pep_tcr[pep] for pep in pep_tcr], reverse=True)
    print(tcr_nums)
    plt.bar(range(len(tcr_nums)), tcr_nums)
    plt.ylabel('TCRs per peptide')
    plt.xlabel('peptide index')
    plt.title(title)
    plt.show()
    pass


w = 'weizmann_pairs.txt'
c = 'cancer_pairs.txt'

# length_dist(w, 'tcr', 'TCR length distribution in Weizmann data')
# length_dist(w, 'pep', 'Peptide length distribution in Weizmann data')
# length_dist(c, 'tcr', 'TCR length distribution in cancer data')
# length_dist(c, 'pep', 'Peptide length distribution in cancer data')
# amino_acids_distribution(w, 'tcr', 'Animo acids distribution in Weizmann TCRs')
# amino_acids_distribution(w, 'pep', 'Animo acids distribution in Weizmann peptides')
# amino_acids_distribution(c, 'tcr', 'Animo acids distribution in cancer TCRs')
# amino_acids_distribution(c, 'pep', 'Animo acids distribution in cancer peptides')
# amino_corr_map(w, 'tcr', 'Weizmann TCRs transition matrix')
# amino_corr_map(w, 'pep', 'Weizmann Peptides transition matrix')
# amino_corr_map(c, 'tcr', 'Cancer TCRs transition matrix')
# amino_corr_map(c, 'pep', 'Cancer Peptides transition matrix')
# tcr_per_pep_dist(w, 'Number of TCR per peptide distribution in Weizmann data')
# tcr_per_pep_dist(c, 'Number of TCR per peptide distribution in cancer data')

'''
print(num_of_peps('weizmann_pairs.txt'))
print(num_of_peps('shugay_pairs.txt'))
print(num_of_peps('cancer_pairs.txt'))
print()
print(num_of_tcrs('weizmann_pairs.txt'))
print(num_of_tcrs('shugay_pairs.txt'))
print(num_of_tcrs('cancer_pairs.txt'))
print()
print(tcr_per_pep('weizmann_pairs.txt'))
print(tcr_per_pep('shugay_pairs.txt'))
print(tcr_per_pep('cancer_pairs.txt'))
'''
