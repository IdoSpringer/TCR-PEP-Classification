import pickle


pep_tcr_dict = {}
with open('SearchTable-2018-11-25 12_15_59.135.tsv', 'r') as file:
    next(file)
    for line in file:
        data = line.split("\t")
        tcr_type = data[1]
        if tcr_type == 'TRB':
            tcr_beta = data[2]
            peptide = data[9]
            if not (tcr_beta == '' or peptide == ''):
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
        # print(len(pep_tcr_dict[peptide]))

print(large_peps)
print(len(large_peps))


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