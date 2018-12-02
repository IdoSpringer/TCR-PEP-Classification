import matplotlib.pyplot as plt
import numpy as np


weizmann = "Weizmann complete database.csv"
shugay = "Shugay complete database.tsv"

def tcr_per_peptide_w():
    peptides = {}
    with open(weizmann, 'r') as data:
        next(data)
        for line in data:
            line = line.split(',')
            print(line)
            cdr_beta = line[2]
            if cdr_beta == 'NA':
                continue
            peptide = line[12]
            if peptide == 'NA':
                continue
            try:
                peptides[peptide] += 1
            except KeyError:
                peptides[peptide] = 1
    list = sorted(peptides, key=lambda k: peptides[k], reverse=True)
    return list, peptides

def tcr_per_peptide_s():
    peptides = {}
    with open(shugay, 'r') as data:
        next(data)
        for line in data:
            line = line.split('\t')
            print(line)
            cdr_type = line[1]
            if cdr_type != "TRB":
                continue
            cdr_beta = line[2]
            peptide = line[9]
            try:
                peptides[peptide] += 1
            except KeyError:
                peptides[peptide] = 1
    list = sorted(peptides, key=lambda k: peptides[k], reverse=True)
    return list, peptides


#list, peptides = tcr_per_peptide_w()
list, peptides = tcr_per_peptide_s()

with open("20tcr_per_peptide_shugay.csv", 'w+') as file:
    file.write('"Number of TCR","Peptide"'+'\n')
    for peptide in list[:20]:
        file.write('"' + str(peptides[peptide]) + '"' + "," + '"'+peptide+'"' + '\n')
