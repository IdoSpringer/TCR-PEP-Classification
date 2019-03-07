
# Shugay pairs with score 3 (safest)

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
            score = int(line[-1])
            if score == 3:
                with open(out, 'a+') as file:
                    file.write(tcr_beta + '\t' + peptide + '\n')
                    print(tcr_beta, peptide)

take_pairs_s('full_shugay.tsv', 'safe_shugay_pairs.txt')

