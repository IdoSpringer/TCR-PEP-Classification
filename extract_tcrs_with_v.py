import csv
import os

# extract from McPAS-TCR

read = 'McPAS-TCR.csv'
filename = 'McPAS-TCR_with_V'


def extract_mcpas(read, filename):
    with open(read, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            # print(line)
            tcr_beta = line[1]
            v_beta_gene = line[20]
            peptide = line[11]
            if not any(k == 'NA' for k in [tcr_beta, v_beta_gene, peptide]):
                print(tcr_beta, v_beta_gene, peptide)
                with open(filename, 'a+') as file2:
                    file2.write('\t'.join([tcr_beta, v_beta_gene, peptide]) + '\n')


path = 'tcrgp_training_data'
filename = 'TCRGP_with_V'


def extract_tcrgp(path, filename):
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            print(file)
            with open(os.path.join(directory, file), mode='r') as infile:
                infile.readline()
                reader = csv.reader(infile)
                for line in reader:
                    print(line)
                    tcr_beta = line[5]
                    v_beta_gene = line[3]
                    peptide = line[0]
                    if not any(k == 'NA' for k in [tcr_beta, v_beta_gene, peptide]):
                        print(tcr_beta, v_beta_gene, peptide)
                        with open(filename, 'a+') as file2:
                            file2.write('\t'.join([tcr_beta, v_beta_gene, peptide]) + '\n')


def extract_negs_tcrgp(path, filename):
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            print(file)
            with open(os.path.join(directory, file), mode='r') as infile:
                infile.readline()
                reader = csv.reader(infile)
                for line in reader:
                    print(line)
                    tcr_beta = line[5]
                    v_beta_gene = line[3]
                    peptide = line[0]
                    if (not any(k == 'NA' for k in [tcr_beta, v_beta_gene, peptide])) and peptide == 'none':
                        print(tcr_beta, v_beta_gene, peptide)
                        with open(filename, 'a+') as file2:
                            file2.write('\t'.join([tcr_beta, v_beta_gene, peptide]) + '\n')


# extract_tcrgp(path, filename)
# extract_negs_tcrgp(path, 'TCRGP_negs_with_v')