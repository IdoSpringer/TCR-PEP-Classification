# Find TCR binding tocancer peptides in all datasets.


def get_cancer_tcrs(cancer_data):
    c_tcrs = []
    with open(cancer_data, 'r') as file:
        for line in file:
            tcr = line.split('\t')[0]
            c_tcrs.append(tcr)
    return set(c_tcrs)


def get_cancer_peps(cancer_data):
    c_peps = []
    with open(cancer_data, 'r') as file:
        for line in file:
            pep = line.strip().split('\t')[-1]
            c_peps.append(pep)
    return set(c_peps)


def extract_cancer_tcrs(data, c_peps):
    c_tcrs = []
    with open(data, 'r') as file:
        for line in file:
            tcr, pep = line.strip().split('\t')
            if pep in c_peps:
                c_tcrs.append(tcr)
    return set(c_tcrs)


def make_c_tcrs_file(filename):
    c_tcrs_in_c = get_cancer_tcrs('pair_sampling/pairs_data/cancer_pairs.txt')
    c_peps = get_cancer_peps('pair_sampling/pairs_data/cancer_pairs.txt')
    c_tcrs_in_w = extract_cancer_tcrs('pair_sampling/pairs_data/weizmann_pairs.txt', c_peps)
    c_tcrs_in_s = extract_cancer_tcrs('pair_sampling/pairs_data/shugay_pairs.txt', c_peps)
    print(len(c_tcrs_in_c), len(c_tcrs_in_w), len(c_tcrs_in_s))
    c_tcrs = set(list(c_tcrs_in_c) + list(c_tcrs_in_w) + list(c_tcrs_in_s))
    print(len(c_tcrs))
    with open(filename, 'a+') as file:
        for c_tcr in c_tcrs:
            file.write(c_tcr + '\n')
    pass

# make_c_tcrs_file('cancer_tcrs_for_shirit')

def make_safe_c_tcrs_file(filename):
    c_tcrs_in_c = get_cancer_tcrs('pair_sampling/pairs_data/cancer_pairs.txt')
    c_peps = get_cancer_peps('pair_sampling/pairs_data/cancer_pairs.txt')
    c_tcrs_in_w = extract_cancer_tcrs('pair_sampling/pairs_data/weizmann_pairs.txt', c_peps)
    c_tcrs_in_s = extract_cancer_tcrs('safe_shugay_pairs.txt', c_peps)
    print(len(c_tcrs_in_c), len(c_tcrs_in_w), len(c_tcrs_in_s))
    c_tcrs = set(list(c_tcrs_in_c) + list(c_tcrs_in_w) + list(c_tcrs_in_s))
    print(len(c_tcrs))
    with open(filename, 'a+') as file:
        for c_tcr in c_tcrs:
            file.write(c_tcr + '\n')
    pass

# make_safe_c_tcrs_file('safe_cancer_tcrs_for_shirit')



def get_cancer_pairs(cancer_data):
    pairs = []
    with open(cancer_data, 'r') as file:
        for line in file:
            tcr, pep = line.strip().split('\t')
            pair = (tcr, pep)
            pairs.append(pair)
    return set(pairs)


def extract_cancer_pairs(data, c_peps):
    pairs = []
    with open(data, 'r') as file:
        for line in file:
            tcr, pep = line.strip().split('\t')
            if pep in c_peps:
                pair = (tcr, pep)
                pairs.append(pair)
    return set(pairs)


def make_c_pairs_file(filename):
    c_pairs_in_c = get_cancer_pairs('pair_sampling/pairs_data/cancer_pairs.txt')
    c_peps = get_cancer_peps('pair_sampling/pairs_data/cancer_pairs.txt')
    c_pairs_in_w = extract_cancer_pairs('pair_sampling/pairs_data/weizmann_pairs.txt', c_peps)
    c_pairs_in_s = extract_cancer_pairs('pair_sampling/pairs_data/shugay_pairs.txt', c_peps)
    print(len(c_pairs_in_c), len(c_pairs_in_w), len(c_pairs_in_s))
    c_pairs = set(list(c_pairs_in_c) + list(c_pairs_in_w) + list(c_pairs_in_s))
    print(len(c_pairs))
    with open(filename, 'a+') as file:
        for pair in c_pairs:
            print(pair)
            tcr, pep = pair
            file.write(tcr + '\t' + pep + '\n')
    pass

# make_c_pairs_file('extended_cancer_pairs.txt')


def make_safe_c_pairs_file(filename):
    c_pairs_in_c = get_cancer_pairs('pair_sampling/pairs_data/cancer_pairs.txt')
    c_peps = get_cancer_peps('pair_sampling/pairs_data/cancer_pairs.txt')
    c_pairs_in_w = extract_cancer_pairs('pair_sampling/pairs_data/weizmann_pairs.txt', c_peps)
    c_pairs_in_s = extract_cancer_pairs('safe_shugay_pairs.txt', c_peps)
    print(len(c_pairs_in_c), len(c_pairs_in_w), len(c_pairs_in_s))
    c_pairs = set(list(c_pairs_in_c) + list(c_pairs_in_w) + list(c_pairs_in_s))
    print(len(c_pairs))
    with open(filename, 'a+') as file:
        for pair in c_pairs:
            print(pair)
            tcr, pep = pair
            file.write(tcr + '\t' + pep + '\n')
    pass


# make_safe_c_pairs_file('safe_extended_cancer_pairs.txt')
