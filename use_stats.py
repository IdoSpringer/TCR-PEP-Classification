import pair_sampling.pairs_data.stats as st


w = 'pair_sampling/pairs_data/weizmann_pairs.txt'
s = 'pair_sampling/pairs_data/shugay_pairs.txt'
exc = 'extended_cancer_pairs.txt'

# st.length_dist(w, 'tcr', 'TCR length distribution in McPAS-TCR')
# st.length_dist(s, 'tcr', 'TCR length distribution in VDJdb')

# st.amino_acids_distribution(w, 'tcr', 'Amino acids distribution in McPAS-TCR')
# st.amino_acids_distribution(w, 'tcr', 'Amino acids distribution in VDJdb')

# st.amino_corr_map(w, 'tcr', 'Amino acids correlation map in McPAS-TCR')
# st.amino_corr_map(w, 'tcr', 'Amino acids correlation map in VDJdb')

# st.tcr_per_pep_dist(w, 'Number of TCRs per peptide distribution in McPAS-TCR')
# st.tcr_per_pep_dist(w, 'Number of TCRs per peptide distribution in VDJdb')


# print(st.num_of_peps(w))
# print(st.num_of_peps(s))
# print(num_of_peps('cancer_pairs.txt'))
# print()
# print(st.num_of_tcrs(w))
# print(st.num_of_tcrs(s))
# print(num_of_tcrs('cancer_pairs.txt'))
# print()
# print(st.tcr_per_pep(w))
# print(st.tcr_per_pep(s))
# print(tcr_per_pep('cancer_pairs.txt'))


def num_of_peps_iedb(pair_file):
    peps = set()
    l = 0
    with open(pair_file, 'r') as file:
        for line in file:
            l += 1
            pep = line.strip().split('\t')[0]
            peps.add(pep)
    num_peps = len(peps)
    print(num_peps, l)
    return num_peps / l


def num_of_tcrs_iedb(pair_file):
    tcrs = set()
    l = 0
    with open(pair_file, 'r') as file:
        for line in file:
            l += 1
            tcr = line.strip().split('\t')[1]
            tcrs.add(tcr)
    num_tcrs = len(tcrs)
    print(num_tcrs, l)
    return num_tcrs / l


def tcr_per_pep_iedb(pair_file):
    pep_tcr = {}
    with open(pair_file, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            tcr = line[1]
            pep = line[0]
            try:
                pep_tcr[pep] += 1
            except KeyError:
                pep_tcr[pep] = 1
    count = 0
    for pep in pep_tcr:
        if pep_tcr[pep] >= 10:
            count += 1
    for pep in pep_tcr:
        if pep_tcr[pep] >= 10:
            print(pep, pep_tcr[pep])
    return count


m = 'netTCR/parameters/iedb_mira_pos_uniq.txt'

print(num_of_peps_iedb(m))
print()
print(num_of_tcrs_iedb(m))
print()
tcr_per_pep_iedb(m)


