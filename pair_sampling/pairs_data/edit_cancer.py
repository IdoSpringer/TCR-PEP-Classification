

def make_cancer_file(cancer_file, output):
    pep_tcr = {}
    with open(cancer_file, 'r') as file:
        for line in file:
            tcr, pep = line.strip().split()
            try:
                pep_tcr[pep].append(tcr)
            except KeyError:
                pep_tcr[pep] = [tcr]
    with open(output, 'a+') as out:
        for pep in pep_tcr:
            if len(pep_tcr[pep]) > 10:
                for tcr in pep_tcr[pep]:
                    print(tcr, pep)
                    out.write(tcr + '\t' + pep + '\n')

make_cancer_file('cancer_pairs.txt', 'cancer_10tcr.txt')