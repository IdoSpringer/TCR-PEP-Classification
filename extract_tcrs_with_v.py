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


def extract_mcpas_tcrs(read, filename):
    with open(read, 'r') as file:
        reader = csv.reader(file)
        try:
            for line in reader:
                # print(line)
                tcr_beta = line[1]
                if not any(k == 'NA' for k in [tcr_beta]):
                    tcr_beta = tcr_beta[3:-1]
                    print(tcr_beta)
                    with open(filename, 'a+') as file2:
                        file2.write(tcr_beta + '\n')
        except UnicodeDecodeError:
            pass


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


def extract_negs_tcrgp_tcrs(path, filename):
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            print(file)
            with open(os.path.join(directory, file), mode='r') as infile:
                infile.readline()
                reader = csv.reader(infile)
                for line in reader:
                    tcr_beta = line[5]
                    peptide = line[0]
                    if (not any(k == 'NA' for k in [tcr_beta])) and peptide == 'none':
                        tcr_beta = tcr_beta[3:-1]
                        print(tcr_beta)
                        with open(filename, 'a+') as file2:
                            file2.write(tcr_beta + '\n')



# extract_tcrgp(path, filename)
# extract_negs_tcrgp(path, 'TCRGP_negs_with_v')

# extract_mcpas_tcrs(read, 'McPAS-TCR_no_cas')
# extract_negs_tcrgp_tcrs('TCRGP/training_data', 'TCRGP_negs_no_cas')