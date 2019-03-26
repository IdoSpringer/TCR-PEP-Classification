import torch
import sys
import os
import matplotlib.pyplot as plt
import load_data_subs as ds
import tcr_ae_pep_lstm_train as tr


def main(argv):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}

    # Load data
    pairs_file = 'pair_sampling/pairs_data/weizmann_pairs.txt'
    if argv[-1] == 'cancer':
        pairs_file = 'pair_sampling/pairs_data/cancer_pairs.txt'
    if argv[-1] == 'shugay':
        pairs_file = 'pair_sampling/pairs_data/shugay_pairs.txt'
    if argv[-1] == 'ex_cancer':
        pairs_file = 'extended_cancer_pairs.txt'
    if argv[-1] == 'exs_cancer':
        pairs_file = 'safe_extended_cancer_pairs.txt'
    if argv[-1] == 'exnos_cancer':
        pairs_file = 'no_shugay_extended_cancer_pairs.txt'

    subsamples = ds.get_subsamples(pairs_file, 1000)
    all_pairs = subsamples[-1]
    iteration = 0
    for subsample in subsamples:
        iteration += 1
        train, test = ds.load_data(subsample, all_pairs)
        # Set all parameters and program arguments
        device = argv[2]
        args = {}
        args['train_auc_file'] = argv[3] + '_' + str(iteration)
        args['test_auc_file'] = argv[4] + '_' + str(iteration)
        args['ae_file'] = argv[1]
        params = {}
        params['lr'] = 1e-3
        params['wd'] = 1e-5
        params['epochs'] = 200
        params['emb_dim'] = 10
        params['enc_dim'] = 30
        params['dropout'] = 0.1
        params['train_ae'] = True
        # Load autoencoder params
        checkpoint = torch.load(args['ae_file'])
        params['max_len'] = checkpoint['max_len']
        params['batch_size'] = checkpoint['batch_size']
        # train
        train_tcrs, train_peps, train_signs = tr.get_lists_from_pairs(train, params['max_len'])
        train_batches = tr.get_batches(train_tcrs, train_peps, train_signs, tcr_atox, pep_atox, params['batch_size'],
                                    params['max_len'])
        # test
        test_tcrs, test_peps, test_signs = tr.get_lists_from_pairs(test, params['max_len'])
        test_batches = tr.get_batches(test_tcrs, test_peps, test_signs, tcr_atox, pep_atox, params['batch_size'],
                                   params['max_len'])
        # Train the model
        model = tr.train_model(train_batches, test_batches, device, args, params)


def max_auc(auc_file):
    with open(auc_file, 'r') as file:
        aucs = []
        for line in file:
            aucs.append(float(line.strip()))
        max_auc = max(aucs)
        return max_auc


def subsamples_auc_graph(key, dir):
    directory = os.fsencode(dir)
    aucs = []
    iterations = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith('ae_' + key + '_test'):
            iteration = int(filename.split('_')[-1])
            auc = max_auc(dir + '/' + filename)
            print(iteration, auc)
            iterations.append(iteration)
            aucs.append(auc)
    aucs = [auc for _, auc in sorted(zip(iterations, aucs))]
    iterations = sorted(iterations)
    plt.plot(iterations, aucs)
    if key == 'w':
        plt.title('Autoencoder model AUC score on sub-samples of McPAS-TCR data')
    elif key == 's':
        plt.title('Autoencoder model AUC score on sub-samples of VDJdb data')
    plt.xlabel('Number of TCR-peptide pairs / 1000')
    plt.xticks(iterations)
    plt.ylabel('best AUC score')
    plt.show()
    pass


if __name__ == '__main__':
    # main(sys.argv)
    subsamples_auc_graph('w', 'subsamples_auc')
    subsamples_auc_graph('s', 'subsamples_auc')
    pass
