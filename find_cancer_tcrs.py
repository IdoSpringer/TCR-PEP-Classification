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


make_c_tcrs_file('cancer_tcrs_for_shirit')