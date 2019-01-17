import pickle
import cProfile, pstats, io


# Profiling
# pr = cProfile.Profile()
# pr.enable()

pep_tcr_dict = {}
with open('McPAS-TCR.csv', 'r') as file:
    next(file)
    for line in file:
        data = line.split(",")
        tcr_beta = data[2]
        peptide = data[3]
        if not (tcr_beta == 'NA' or peptide == 'NA'):
            try:
                pep_tcr_dict[peptide].append(tcr_beta)
            except KeyError:
                pep_tcr_dict[peptide] = []
                pep_tcr_dict[peptide].append(tcr_beta)


print(len(list(pep_tcr_dict.keys())))

list1 = sorted(pep_tcr_dict, key=lambda k: len(pep_tcr_dict[k]), reverse=True)

large_peps = []
for peptide in pep_tcr_dict:
    if len(pep_tcr_dict[peptide]) >= 100:
        large_peps.append(peptide)

print(large_peps)
print(len(large_peps))

# pr.disable()
# pr.dump_stats('profiling_1pep_learning')
'''
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
with open('profiling_1pep_learning', 'w') as stream:
    stats = pstats.Stats('path/to/input', stream=stream)
    stats.print_stats()
'''
# p = pstats.Stats('profiling_1pep_learning')
# p.sort_stats('cumulative').print_stats(10)


'''
print(list1[:20])

for peptide in list1:
    print(len(pep_tcr_dict[peptide]))
'''

# Make original data
'''
train_peptides = ['EAAGIGILTV', 'GLCTLVAML', 'RAKFKQLL', 'NLVPMVATV', 'KAFSPEVIPMF',
            'ATDALMTGY', 'TPRVTGGGAM', 'KRWIILGLNK', 'YSEHPTFTSQY', 'GILGFVFTL',
            'FPRPWLHGL', 'HPKVSSEVHI', 'LLWNGPMAV', 'VTEHDTLLY', 'LPRRSGAAGA']


for peptide in list(dict.keys()):
    if not peptide in peptides:
        pep_tcr_dict.pop(peptide, None)


for peptide in list(dict.keys()):
    print(peptide, ": ")
    print(pep_tcr_dict[peptide])

# pickle.dump(dict, open("save_pep_train_dict.pickle", 'wb'))
'''