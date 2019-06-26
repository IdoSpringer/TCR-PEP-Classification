import random
import numpy as np
import csv
import os
import sklearn.model_selection as skl


def read_data(csv_file, file_key):
    with open(csv_file, 'r', encoding='unicode_escape') as file:
        file.readline()
        if file_key == 'mcpas':
            reader = csv.reader(file)
        elif file_key == 'vdjdb':
            reader = csv.reader(file, delimiter='\t')
        tcrs = set()
        peps = set()
        all_pairs = []
        for line in reader:
            if file_key == 'mcpas':
                tcr, pep, protein = line[1], line[11], line[9]
            elif file_key == 'vdjdb':
                tcr, pep, protein = line[2], line[9], line[10]
                if line[1] != 'TRB':
                    continue
            # Proper tcr and peptides, human MHC
            if any(att == 'NA' for att in [tcr, pep, protein]):
                continue
            if any(key in tcr + pep for key in ['#', '*', 'b', 'f', 'y', '~', 'O', '/']):
                continue
            tcrs.add(tcr)
            peps.add((pep, protein))
            all_pairs.append((tcr, (pep, protein)))
    train_pairs, test_pairs = train_test_split(all_pairs)
    return all_pairs, train_pairs, test_pairs


def train_test_split(all_pairs):
    train_pairs = []
    test_pairs = []
    for pair in all_pairs:
        # 80% train, 20% test
        p = np.random.binomial(1, 0.8)
        if p == 1:
            train_pairs.append(pair)
        else:
            test_pairs.append(pair)
    return train_pairs, test_pairs


def positive_examples(pairs):
    examples = []
    for pair in pairs:
        tcr, (pep, protein) = pair
        examples.append((tcr, pep, 'p'))
    return examples


def negative_examples(pairs, all_pairs, size):
    examples = []
    i = 0
    # Get tcr and peps lists
    tcrs = [tcr for (tcr, (pep, protein)) in pairs]
    peps_proteins = [(pep, protein) for (tcr, (pep, protein)) in pairs]
    while i < size:
        pep_protein = random.choice(peps_proteins)
        for j in range(5):
            tcr = random.choice(tcrs)
            tcr_pos_pairs = [pair for pair in all_pairs if pair[0] == tcr]
            tcr_proteins = [protein for (tcr, (pep, protein)) in tcr_pos_pairs]
            pep, protein = pep_protein
            attach = protein in tcr_proteins
            if attach is False:
                if (tcr, pep, 'n') not in examples:
                    examples.append((tcr, pep, 'n'))
                    i += 1
    return examples


def read_negs(dir):
    neg_tcrs = []
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            with open(dir + '/' + filename, 'r') as csv_file:
                csv_file.readline()
                csv_ = csv.reader(csv_file)
                for row in csv_:
                    if row[1] == 'control':
                        tcr = row[-1]
                        neg_tcrs.append(tcr)
    train, test, _, _ = skl.train_test_split(neg_tcrs, neg_tcrs, test_size=0.2)
    return train, test


def negative_naive_examples(pairs, all_pairs, size, tcrgp_negs):
    examples = []
    i = 0
    # Get tcr and peps lists
    peps_proteins = [(pep, protein) for (tcr, (pep, protein)) in pairs]
    while i < size:
        pep_protein = random.choice(peps_proteins)
        for j in range(5):
            tcr = random.choice(tcrgp_negs)
            pep, protein = pep_protein
            attach = (tcr, (pep, protein)) in all_pairs
            if attach is False:
                if (tcr, pep, 'n') not in examples:
                    examples.append((tcr, pep, 'n'))
                    i += 1
    return examples


def get_examples(pairs_file, key, naive=False):
    all_pairs, train_pairs, test_pairs = read_data(pairs_file, key)
    train_pos = positive_examples(train_pairs)
    test_pos = positive_examples(test_pairs)
    if naive:
        neg_train, neg_test = read_negs('TCRGP/training_data')
        train_neg = negative_naive_examples(train_pairs, all_pairs, len(train_pos), neg_train)
        test_neg = negative_naive_examples(test_pairs, all_pairs, len(test_pos), neg_train)
    else:
        train_neg = negative_examples(train_pairs, all_pairs, len(train_pos))
        test_neg = negative_examples(test_pairs, all_pairs, len(test_pos))
    return train_pos, train_neg, test_pos, test_neg


def load_data(pairs_file, key, naive):
    train_pos, train_neg, test_pos, test_neg = get_examples(pairs_file, key, naive)
    train = train_pos + train_neg
    random.shuffle(train)
    test = test_pos + test_neg
    random.shuffle(test)
    return train, test


def check(file, key, naive):
    train, test = load_data(file, key, naive)
    print(train)
    print(test)
    print(len(train))
    print(len(test))


# check('McPAS-TCR.csv', 'mcpas')
