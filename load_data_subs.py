import random
from random import shuffle, sample
import numpy as np


def get_subsamples(pairs_file, j):
    with open(pairs_file, 'r') as file:
        all_pairs = []
        for line in file:
            tcr, pep = line.strip().split('\t')
            if '*' in tcr or '*' in pep:
                continue
            if '/' in pep:
                continue
            all_pairs.append((tcr, pep))
    num_examples = len(all_pairs)
    i = 0
    subsamples = [[]]
    while i < num_examples:
        j = min(j, num_examples - i)
        new = sample(all_pairs, j)
        subsample = subsamples[-1] + new
        shuffle(subsample)
        for example in new:
            all_pairs.remove(example)
        subsamples.append(subsample)
        i += j
    assert i == num_examples
    return subsamples[1:]


def train_test_split(pairs):
    train_pairs = []
    test_pairs = []
    for pair in pairs:
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
        tcr, pep = pair
        weight = 1
        examples.append((tcr, pep, 'p', weight))
    return examples


def negative_examples(pairs, all_pairs, size):
    examples = []
    i = 0
    # Get tcr and peps lists
    tcrs = [tcr for (tcr, pep) in pairs]
    peps = [pep for (tcr, pep) in pairs]
    while i < size:
        pep = random.choice(peps)
        for j in range(5):
            tcr = random.choice(tcrs)
            attach = (tcr, pep) in all_pairs
            if attach is False:
                weight = 1
                if (tcr, pep, 'n', weight) not in examples:
                    examples.append((tcr, pep, 'n', weight))
                    i += 1
    return examples


def get_examples(subsample, all_pairs):
    train_pairs, test_pairs = train_test_split(subsample)
    train_pos = positive_examples(train_pairs)
    train_neg = negative_examples(train_pairs, all_pairs, len(train_pos))
    test_pos = positive_examples(test_pairs)
    test_neg = negative_examples(test_pairs, all_pairs, len(test_pos))
    return train_pos, train_neg, test_pos, test_neg


def load_data(subsample, all_pairs):
    train_pos, train_neg, test_pos, test_neg = get_examples(subsample, all_pairs)
    train = train_pos + train_neg
    random.shuffle(train)
    test = test_pos + test_neg
    random.shuffle(test)
    return train, test


def check():
    pairs_file = 'pairs_data/weizmann_pairs.txt'
    train, test = load_data(pairs_file)
    print(len(train), train)
    print(len(test), test)


# check()
