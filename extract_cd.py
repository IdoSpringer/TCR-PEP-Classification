import csv

w = 'McPAS-TCR.csv'


def take_pairs_w(w, out):
    with open(w, 'r', encoding='utf-8', errors='ignore') as data_csv:
        next(data_csv)
        reader = csv.reader(data_csv, delimiter=',', quotechar='"')
        index = 0
        for line in reader:
            tcr_beta = line[1]
            if tcr_beta == 'NA':
                continue
            peptide = line[11]
            if peptide == 'NA':
                continue
            cd = line[15]
            index += 1
            with open(out, 'a+') as file:
                file.write('\t'.join([tcr_beta, peptide, cd]) + '\n')
            print(index, tcr_beta, peptide, cd)


out = 'McPAS-with_CD'

take_pairs_w(w, out)
