import matplotlib.pyplot as plt
import compare_data_stats as cmp
from Kidera import kidera
import numpy as np


w = 'McPAS-TCR_with_V'
t = 'TCRGP_with_V'
nt = 'TCRGP_negs_with_V'


def kidera_hists(data1, data2):
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
    fig = plt.figure()
    for i in range(10):
        ax = fig.add_subplot(2, 5, i+1)
        a = factor_observations1[i]
        b = factor_observations2[i]
        weights1 = np.ones_like(a) / float(len(a))
        weights2 = np.ones_like(b) / float(len(b))
        bins = np.linspace(-1.0, 1.0, 10)
        plot2 = ax.hist([t + 0.1 for t in b], weights=weights2, bins=[bin + 0.1 for bin in bins],
                        color='salmon', label='TCRGP', width=0.1)
        plot1 = ax.hist(a, weights=weights1, bins=bins,
                        color='dodgerblue', label='McPAS', width=0.1)

        ax.set_title('Kidera ' + str(i+1) + ' factor histogram')
        ax.legend()
    fig.tight_layout()
    plt.show()


kidera_hists(w, nt)


