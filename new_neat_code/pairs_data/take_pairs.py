import csv

def take_pairs_w(w, out):
    with open(w, 'r') as data_csv:
        # next(data)
        reader = csv.reader(data_csv, delimiter=',', quotechar='"')
        for line in reader:
            tcr_beta = line[2]
            if tcr_beta == 'NA':
                continue
            peptide = line[12]
            if peptide == 'NA':
                continue
            with open(out, 'a+') as file:
                file.write(tcr_beta + '\t' + peptide + '\n')
                print(tcr_beta, peptide)


# take_pairs_w('Weizmann complete database.csv', 'weizmann_pairs1.txt')


def take_pairs_s(s, out):
    with open(s, 'r') as data:
        next(data)
        for line in data:
            line = line.split('\t')
            cdr_type = line[1]
            if cdr_type != "TRB":
                continue
            tcr_beta = line[2]
            peptide = line[9]
            with open(out, 'a+') as file:
                file.write(tcr_beta + '\t' + peptide + '\n')
                print(tcr_beta, peptide)

# take_pairs_s('Shugay complete database.tsv', 'shugay_pairs.txt')
