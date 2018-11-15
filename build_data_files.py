import pickle

dict = {}
with open('McPAS-TCR.csv', 'r') as file:
    next(file)
    for line in file:
        data = line.split(",")
        tcr_beta = data[2]
        peptide = data[3]
        if not (tcr_beta == 'NA' or peptide == 'NA'):
            try:
                dict[peptide].append(tcr_beta)
            except KeyError:
                dict[peptide] = []
                dict[peptide].append(tcr_beta)



print(len(list(dict.keys())))

list1 = sorted(dict, key=lambda k: len(dict[k]), reverse=True)
#  print(list1[:20])

'''
for peptide in list:
    print(len(dict[peptide]))
'''

peptides = ['EAAGIGILTV', 'GLCTLVAML', 'RAKFKQLL', 'NLVPMVATV', 'KAFSPEVIPMF',
            'ATDALMTGY', 'TPRVTGGGAM', 'KRWIILGLNK', 'YSEHPTFTSQY', 'GILGFVFTL',
            'FPRPWLHGL', 'HPKVSSEVHI', 'LLWNGPMAV', 'VTEHDTLLY', 'LPRRSGAAGA']


for peptide in list(dict.keys()):
    if not peptide in peptides:
        dict.pop(peptide, None)


for peptide in list(dict.keys()):
    print(peptide, ": ")
    print(dict[peptide])

pickle.dump(dict, open("save_pep_train_dict.pickle", 'wb'))
