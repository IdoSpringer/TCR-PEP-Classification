import matplotlib.pyplot as plt
import csv
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


def tcr_per_peptide_u():
    peptides = {}
    with open(weizmann, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for line in csv_reader:
            cdr_beta = line[2]
            if cdr_beta == 'NA':
                continue
            peptide = line[12]
            print(cdr_beta, peptide, 'w')
            if peptide == 'NA':
                continue
            try:
                peptides[peptide] += 1
            except KeyError:
                peptides[peptide] = 1
    with open(shugay, 'r') as data:
        next(data)
        for line in data:
            line = line.split('\t')
            cdr_type = line[1]
            if cdr_type != "TRB":
                continue
            cdr_beta = line[2]
            peptide = line[9]
            print(cdr_beta, peptide, 's')
            try:
                peptides[peptide] += 1
            except KeyError:
                peptides[peptide] = 1
    list = sorted(peptides, key=lambda k: peptides[k], reverse=True)
    return list, peptides
    pass


# Get number of TCR per peptide
'''
# list, peptides = tcr_per_peptide_w()
# list, peptides = tcr_per_peptide_s()
# list, peptides = tcr_per_peptide_u()

with open("20tcr_per_peptide_union.csv", 'w+') as file:
    file.write('"Number of TCR","Peptide"'+'\n')
    for peptide in list[:20]:
        file.write('"' + str(peptides[peptide]) + '"' + "," + '"'+peptide+'"' + '\n')
'''


def length_distribution_w():
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


def length_distribution_s():
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
    plt.xticks(range(7, 27))
    plt.title("TCR and peptide length distribution, Shugay database")
    plt.show()


def length_distribution_u():
    tcr_len = {}
    pep_len = {}
    with open(weizmann, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for line in csv_reader:
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
    with open(shugay, 'r') as data:
        next(data)
        for line in data:
            line = line.split('\t')
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
    plt.xticks(range(7, 21))
    plt.title("TCR and peptide length distribution, Union database")
    plt.show()


# length_distribution_w()
# length_distribution_s()
# length_distribution_u()


'''
# list, peptides = tcr_per_peptide_w()
# list, peptides = tcr_per_peptide_s()
print(list)
print(peptides)
print(len([pep for pep in peptides.keys()]))
print(len([pep for pep in peptides.keys() if peptides[pep] > 100]))
print(len([pep for pep in peptides.keys() if peptides[pep] > 500]))
'''


def pep_and_tcr_per_disease_w():
    diseases_tcr = {}
    diseases_pep = {}
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
            disease = line[5]
            try:
                diseases_tcr[disease].append(cdr_beta)
            except KeyError:
                diseases_tcr[disease] = [cdr_beta]
            try:
                diseases_pep[disease].add(peptide)
            except KeyError:
                diseases_pep[disease] = set()
                diseases_pep[disease].add(peptide)
    list_tcr = sorted(diseases_tcr, key=lambda k: len(diseases_tcr[k]), reverse=True)
    with open('disease_tcr_w.csv', 'w+') as file:
        file.write('"Number of TCR", "Disease"'+'\n')
        for disease in list_tcr[:20]:
            file.write('"'+str(len(diseases_tcr[disease]))+'"'+","+disease+'\n')
    print(disease ,len(diseases_tcr[disease]))
    list_pep = sorted(diseases_pep, key=lambda k: len(diseases_pep[k]), reverse=True)
    with open('disease_pep_w.csv', 'w+') as file:
        file.write('"Number of peptides", "Disease"'+'\n')
        for disease in list_pep[:15]:
            file.write('"' + str(len(diseases_pep[disease])) + '"' + "," + disease + '\n')
        print(disease, len(diseases_pep[disease]))
    return diseases_tcr, diseases_pep


def pep_and_tcr_per_disease_s():
    diseases_tcr = {}
    diseases_pep = {}
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
            disease = line[11]
            try:
                diseases_tcr[disease].append(cdr_beta)
            except KeyError:
                diseases_tcr[disease] = [cdr_beta]
            try:
                diseases_pep[disease].add(peptide)
            except KeyError:
                diseases_pep[disease] = set()
                diseases_pep[disease].add(peptide)
    list_tcr = sorted(diseases_tcr, key=lambda k: len(diseases_tcr[k]), reverse=True)
    for disease in list_tcr[:20]:
        print(disease ,len(diseases_tcr[disease]))
    list_pep = sorted(diseases_pep, key=lambda k: len(diseases_pep[k]), reverse=True)
    for disease in list_pep[:15]:
        print(disease, len(diseases_pep[disease]))
    #print(list_tcr[:20])
    return diseases_tcr, diseases_pep
    #list = sorted(peptides, key=lambda k: peptides[k], reverse=True)

'''
print(pep_and_tcr_per_disease_w())
# print(pep_and_tcr_per_disease_s())
'''