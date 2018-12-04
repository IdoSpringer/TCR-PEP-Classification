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

# Get number of TCR per peptide
'''
list, peptides = tcr_per_peptide_w()
list, peptides = tcr_per_peptide_s()

with open("20tcr_per_peptide_shugay.csv", 'w+') as file:
    file.write('"Number of TCR","Peptide"'+'\n')
    for peptide in list[:20]:
        file.write('"' + str(peptides[peptide]) + '"' + "," + '"'+peptide+'"' + '\n')
'''


def tcr_length_distribution_w():
    tcr_len = {}
    pep_len = {}
    with open(weizmann, 'r') as data:
        next(data)
        for line in data:
            line = line.split(',')
            print(line)
            tcr_beta = line[2]
            if tcr_beta == 'NA':
                continue
            peptide = line[12]
            if peptide == 'NA':
                continue
            try:
                tcr_len[len(tcr_beta)] += 1
            except KeyError:
                tcr_len[len(tcr_beta)] = 1
            try:
                pep_len[len(peptide)] += 1
            except KeyError:
                pep_len[len(peptide)] = 1
    lens_tcr = sorted(list(tcr_len.keys()))
    lens_pep = sorted(list(pep_len.keys()))
    num_len_tcr = [tcr_len[length] for length in lens_tcr]
    num_len_pep = [pep_len[length] for length in lens_pep]
    fig, ax = plt.subplots()
    ax.bar(lens_tcr, num_len_tcr, color='SkyBlue', label='TCR')
    ax.bar(lens_pep, num_len_pep, color='IndianRed', label='peptide')
    ax.legend()
    plt.xticks(lens_tcr)
    plt.title("TCR and peptide length distribution, Weizmann database")
    plt.show()


def tcr_length_distribution_s():
    tcr_len = {}
    pep_len = {}
    with open(shugay, 'r') as data:
        next(data)
        for line in data:
            line = line.split('\t')
            print(line)
            cdr_type = line[1]
            if cdr_type != "TRB":
                continue
            tcr_beta = line[2]
            peptide = line[9]
            if len(peptide) > 26 or len(peptide) < 7:
                continue
            if len(tcr_beta) > 26 or len(tcr_beta) < 7:
                continue
            try:
                tcr_len[len(tcr_beta)] += 1
            except KeyError:
                tcr_len[len(tcr_beta)] = 1
            try:
                pep_len[len(peptide)] += 1
            except KeyError:
                pep_len[len(peptide)] = 1
    lens_tcr = sorted(list(tcr_len.keys()))
    lens_pep = sorted(list(pep_len.keys()))
    num_len_tcr = [tcr_len[length] for length in lens_tcr]
    num_len_pep = [pep_len[length] for length in lens_pep]
    fig, ax = plt.subplots()
    ax.bar(lens_tcr, num_len_tcr, color='SkyBlue', label='TCR')
    ax.bar(lens_pep, num_len_pep, color='IndianRed', label='peptide')
    ax.legend()
    plt.xticks(range(7,27))
    plt.title("TCR and peptide length distribution, Shugay database")
    plt.show()


tcr_length_distribution_w()
tcr_length_distribution_s()

def pep_length_distribution_w():
    pass

def pep_length_distribution_s():
    pass