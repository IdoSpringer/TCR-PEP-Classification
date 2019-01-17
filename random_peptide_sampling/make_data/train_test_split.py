import pickle
import numpy as np
import csv


# Get train test percent in hanan files
'''
with open('train sequences.pickle', 'rb') as train_file,\
     open('test sequences.pickle', 'rb') as test_file:
    train_list = pickle.load(train_file)
    test_list = pickle.load(test_file)
    for pep in train_list:
        print(len(train_list[pep]), len(test_list[pep]))
        print(len(train_list[pep]) / (len(train_list[pep]) + len(test_list[pep])), 
        len(test_list[pep]) / (len(train_list[pep]) + len(test_list[pep])))

# TRAIN: 80%,   TEST: 20%

'''


# Split Weizmann data
def train_test_split_weizmann():
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
        if len(pep_tcr_dict[peptide]) >= 125:
            large_peps.append(peptide)

    print(large_peps)
    print(len(large_peps))  # 16 peptides

    p = 0.8
    train_peps = {}
    test_peps = {}
    for pep in large_peps:
        train_peps[pep] = []
        test_peps[pep] = []
        train_test_split = np.random.binomial(1, p, size=len(pep_tcr_dict[pep]))
        # print(train_test_split)
        # print(sum(train_test_split) / len(train_test_split))
        for i in range(len(train_test_split)):
            if train_test_split[i]:
                train_peps[pep].append((pep_tcr_dict[pep])[i])
            else:
                test_peps[pep].append((pep_tcr_dict[pep])[i])
    # check
    for pep in train_peps:
        print(len(train_peps[pep]), len(test_peps[pep]))
        print(len(train_peps[pep]) / (len(train_peps[pep]) + len(test_peps[pep])),
              len(test_peps[pep]) / (len(train_peps[pep]) + len(test_peps[pep])))

    pickle.dump(train_peps, open('weizmann_train', 'wb'))
    pickle.dump(test_peps, open('weizmann_test', 'wb'))

    pass


# Split Shugay data
def train_test_split_shugay(shugay_file):
    pep_tcr_dict = {}

    with open(shugay_file, 'r') as file:
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

    print("Num of peptides in file:")
    print(len(list(pep_tcr_dict.keys())))


    large_peps = []
    for peptide in pep_tcr_dict:
        if len(pep_tcr_dict[peptide]) >= 125:
            large_peps.append(peptide)

    print("Peptides with more than 125 tcr:")
    print(large_peps)
    print(len(large_peps))  # 33 peptides


    
    p = 0.8
    train_peps = {}
    test_peps = {}
    for pep in large_peps:
        train_peps[pep] = []
        test_peps[pep] = []
        train_test_split = np.random.binomial(1, p, size=len(pep_tcr_dict[pep]))
        for i in range(len(train_test_split)):
            if train_test_split[i]:
                train_peps[pep].append((pep_tcr_dict[pep])[i])
            else:
                test_peps[pep].append((pep_tcr_dict[pep])[i])
    # check
    for pep in train_peps:
        print(len(train_peps[pep]), len(test_peps[pep]))
        print(len(train_peps[pep]) / (len(train_peps[pep]) + len(test_peps[pep])),
              len(test_peps[pep]) / (len(train_peps[pep]) + len(test_peps[pep])))

    pickle.dump(train_peps, open('shugay_train.pickle', 'wb'))
    pickle.dump(test_peps, open('shugay_test.pickle', 'wb'))

    pass


# Split union data
def train_test_split_union(weizmann_file, shugay_file):
    # peptide-tcr dictionary
    pep_tcr_dict = {}

    # Get weizmann peptides
    with open(weizmann_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for line in csv_reader:
            tcr_beta = line[2]
            peptide = line[12]
            # print(tcr_beta, peptide, 'w')
            if not (tcr_beta == 'NA' or peptide == 'NA'):
                try:
                    pep_tcr_dict[peptide].append(tcr_beta)
                except KeyError:
                    pep_tcr_dict[peptide] = []
                    pep_tcr_dict[peptide].append(tcr_beta)

    # Get Shugay peptides
    with open(shugay_file, 'r') as file:
        # pass headline
        next(file)
        for line in file:
            data = line.split("\t")
            tcr_type = data[1]
            if tcr_type == 'TRB':
                tcr_beta = data[2]
                peptide = data[9]
                # print(tcr_beta, peptide, 's')
                if not (tcr_beta == '' or peptide == ''):
                    try:
                        pep_tcr_dict[peptide].append(tcr_beta)
                    except KeyError:
                        pep_tcr_dict[peptide] = []
                        pep_tcr_dict[peptide].append(tcr_beta)

    print("Num of peptides in file:")
    print(len(list(pep_tcr_dict.keys())))   # 226 peps in union

    large_peps = []
    for peptide in pep_tcr_dict:
        if len(pep_tcr_dict[peptide]) >= 125:
            large_peps.append(peptide)

    print("Peptides with more than 125 tcr:")
    print(large_peps)
    print(len(large_peps))  # 42 peps in union

    print()
    len_list = []
    for pep in large_peps:
        len_list.append(len(pep_tcr_dict[pep]))
    print(sorted(len_list, reverse=True))


    p = 0.8
    train_peps = {}
    test_peps = {}
    for pep in large_peps:
        train_peps[pep] = []
        test_peps[pep] = []
        train_test_split = np.random.binomial(1, p, size=len(pep_tcr_dict[pep]))
        for i in range(len(train_test_split)):
            if train_test_split[i]:
                train_peps[pep].append((pep_tcr_dict[pep])[i])
            else:
                test_peps[pep].append((pep_tcr_dict[pep])[i])
    # check
    for pep in train_peps:
        print(len(train_peps[pep]), len(test_peps[pep]))
        print(len(train_peps[pep]) / (len(train_peps[pep]) + len(test_peps[pep])),
              len(test_peps[pep]) / (len(train_peps[pep]) + len(test_peps[pep])))

    pickle.dump(train_peps, open('union_train.pickle', 'wb'))
    pickle.dump(test_peps, open('union_test.pickle', 'wb'))


# train_test_split_union('Weizmann complete database.csv', 'Shugay complete database.tsv')




