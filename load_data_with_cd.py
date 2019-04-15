import random
import numpy as np


def read_data(pairs_file):
    with open(pairs_file, 'r') as file:
        tcrs = set()
        peps = set()
        all_pairs = []
        for line in file:
            tcr, pep, cd = line.strip().split('\t')
            # print(tcr, pep)
            # Proper tcr and peptides
            if '*' in tcr or '*' in pep:
                continue
            if '/' in pep:
                continue
            tcrs.add(tcr)
            peps.add(pep)
            all_pairs.append((tcr, pep, cd))
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
        tcr, pep, cd = pair
        weight = 1
        examples.append((tcr, pep, cd, 'p', weight))
    return examples


def negative_examples(pairs, all_pairs, size):
    examples = []
    i = 0
    # Get tcr and peps lists
    tcrs = [tcr for (tcr, pep, cd) in pairs]
    peps = [pep for (tcr, pep, cd) in pairs]
    while i < size:
        pep = random.choice(peps)
        for j in range(5):
            tcr = random.choice(tcrs)
            attach = (tcr, pep, 'CD4') in all_pairs\
                     or (tcr, pep, 'CD8') in all_pairs\
                     or (tcr, pep, 'NA') in all_pairs
            if attach is False:
                weight = 1
                if (tcr, pep, 'NEG', 'n', weight) not in examples:
                    examples.append((tcr, pep, 'NEG', 'n', weight))
                    i += 1
    return examples


def get_examples(pairs_file):
    all_pairs, train_pairs, test_pairs = read_data(pairs_file)
    train_pos = positive_examples(train_pairs)
    train_neg = negative_examples(train_pairs, all_pairs, len(train_pos))
    test_pos = positive_examples(test_pairs)
    test_neg = negative_examples(test_pairs, all_pairs, len(test_pos))
    return train_pos, train_neg, test_pos, test_neg


def load_data(pairs_file):
    train_pos, train_neg, test_pos, test_neg = get_examples(pairs_file)
    train = train_pos + train_neg
    random.shuffle(train)
    test = test_pos + test_neg
    random.shuffle(test)
    return train, test


def check():
    pairs_file = 'McPAS-with_CD'
    train, test = load_data(pairs_file)
    print(train)
    print(test)
    print(len(train))
    print(len(test))


# check()
