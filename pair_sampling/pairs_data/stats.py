

def num_of_peps(pair_file):
    peps = set()
    l = 0
    with open(pair_file, 'r') as file:
        for line in file:
            l += 1
            pep = line.strip().split('\t')[-1]
            peps.add(pep)
    num_peps = len(peps)
    print(num_peps, l)
    return num_peps / l


def num_of_tcrs(pair_file):
    tcrs = set()
    l = 0
    with open(pair_file, 'r') as file:
        for line in file:
            l += 1
            tcr = line.strip().split('\t')[0]
            tcrs.add(tcr)
    num_tcrs = len(tcrs)
    print(num_tcrs, l)
    return num_tcrs / l


def tcr_per_pep(pair_file):
    pep_tcr = {}
    with open(pair_file, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            tcr = line[0]
            pep = line[1]
            try:
                pep_tcr[pep].append(tcr)
            except KeyError:
                pep_tcr[pep] = [tcr]
    count = 0
    for pep in pep_tcr:
        if len(pep_tcr[pep]) >= 10:
            count += 1
    return count




print(num_of_peps('weizmann_pairs.txt'))
print(num_of_peps('shugay_pairs.txt'))
print(num_of_peps('cancer_pairs.txt'))
print()
print(num_of_tcrs('weizmann_pairs.txt'))
print(num_of_tcrs('shugay_pairs.txt'))
print(num_of_tcrs('cancer_pairs.txt'))
print()
print(tcr_per_pep('weizmann_pairs.txt'))
print(tcr_per_pep('shugay_pairs.txt'))
print(tcr_per_pep('cancer_pairs.txt'))