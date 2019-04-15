from ae_pep_cd_test_eval import *
from scipy import stats


def main(argv):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}
    args = {}
    args['ae_file'] = 'pad_full_data_autoencoder_model1.pt'
    params = {}
    params['lr'] = 1e-3
    params['wd'] = 1e-5
    params['epochs'] = 200
    params['emb_dim'] = 10
    params['enc_dim'] = 30
    params['dropout'] = 0.1
    params['train_ae'] = False
    # Load autoencoder params
    checkpoint = torch.load(args['ae_file'])
    params['max_len'] = checkpoint['max_len']
    params['batch_size'] = checkpoint['batch_size']
    batch_size = params['batch_size']

    directory = 'test_and_models_with_cd/'
    auc_mat = np.zeros((10, 8))
    for iteration in range(10):
        # load test
        test_file = directory + 'ae_test_w_' + str(iteration)
        model_file = directory + 'ae_model_w_' + str(iteration)
        device = argv[1]
        with open(test_file, 'rb') as fp:
            test = pickle.load(fp)
        # test
        test_tcrs, test_peps, test_signs = get_lists_from_pairs(test, params['max_len'])
        test_batches = get_batches(test_tcrs, test_peps, test_signs, tcr_atox, pep_atox, params['batch_size'],
                                   params['max_len'])
        # load model
        model = AutoencoderLSTMClassifier(params['emb_dim'], device, params['max_len'], 21, params['enc_dim'],
                                          params['batch_size'], args['ae_file'], params['train_ae'])
        trained_model = torch.load(model_file)
        model.load_state_dict(trained_model['model_state_dict'])
        model.eval()
        model = model.to(device)

        peps_pos_probs = {}
        for i in range(len(test_batches)):
            batch = test_batches[i]
            batch_data = test[i * batch_size: (i + 1) * batch_size]
            tcrs, padded_peps, pep_lens, batch_signs = batch
            # Move to GPU
            tcrs = torch.tensor(tcrs).to(device)
            padded_peps = padded_peps.to(device)
            pep_lens = pep_lens.to(device)
            probs = model(tcrs, padded_peps, pep_lens)
            peps = [data[1] for data in batch_data]
            # cd = [data[2] for data in batch_data]
            for pep, prob, sign in zip(peps, probs, batch_signs):
                try:
                    peps_pos_probs[pep].append((prob.item(), sign))
                except KeyError:
                    peps_pos_probs[pep] = [(prob.item(), sign)]
        bins = {}
        for pep in peps_pos_probs:
            num_examples = len(peps_pos_probs[pep])
            bin = int(np.floor(np.log2(num_examples)))
            try:
                bins[bin].extend(peps_pos_probs[pep])
            except KeyError:
                bins[bin] = peps_pos_probs[pep]
        for bin in bins:
            pass
            # print(bin, len(bins[bin]))
        bin_aucs = {}
        for bin in bins:
            try:
                auc = roc_auc_score([p[1] for p in bins[bin]], [p[0] for p in bins[bin]])
                bin_aucs[bin] = auc
                # print(bin, auc)
            except ValueError:
                # print(bin, [p[1] for p in bins[bin]])
                pass
        bin_aucs = sorted(bin_aucs.items())
        print(bin_aucs)
        auc_mat[iteration] = np.array([t[1] for t in bin_aucs])
        pass
    # print(auc_mat)
    means = np.mean(auc_mat, axis=0)
    std = stats.sem(auc_mat, axis=0)
    print(means, std)
    plt.errorbar([j[0] for j in bin_aucs], means, yerr=std)
    plt.xticks([j[0] for j in bin_aucs], [2 ** j[0] for j in bin_aucs])
    plt.xlabel('number of peptide TCRs bins')
    plt.ylabel('averaged AUC score')
    plt.title('AUC per number of TCRs per peptide')
    plt.show()

    '''
    tcr_cd = {}
    for sample in test:
        tcr = sample[0]
        cd = sample[2]
        sign = sample[3]
        if sign is 'p':
            tcr_cd[tcr] = cd
    # print(tcr_cd)
    cd_probs = {}
    for i in range(len(test_batches)):
        batch = test_batches[i]
        batch_data = test[i * batch_size: (i + 1) * batch_size]
        tcrs, padded_peps, pep_lens, batch_signs = batch
        positive_indexes = [i for i in range(batch_size) if batch_signs[i] == 1.0]
        # Move to GPU
        tcrs = torch.tensor(tcrs).to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(tcrs, padded_peps, pep_lens)
        tcrs = [data[0] for data in batch_data]
        for tcr, prob, sign in zip(tcrs, probs, batch_signs):
            try:
                cd_probs[tcr_cd[tcr]].append((prob.item(), sign))
            except KeyError:
                cd_probs[tcr_cd[tcr]] = [(prob.item(), sign)]
    cd_aucs = {}
    for cd in cd_probs:
        try:
            auc = roc_auc_score([p[1] for p in cd_probs[cd]], [p[0] for p in cd_probs[cd]])
            cd_aucs[cd] = auc
            # print(bin, auc)
        except ValueError:
            # print(bin, [p[1] for p in bins[bin]])
            pass
    cd_aucs = sorted(cd_aucs.items())
    print(cd_aucs)
    exit()
    plt.plot([j[0] for j in bin_aucs], [j[1] for j in bin_aucs])
    plt.xticks([j[0] for j in bin_aucs], [2 ** j[0] for j in bin_aucs])
    plt.xlabel('number of peptide TCRs bins')
    plt.ylabel('AUC score')
    plt.title('AUC per number of TCRs per peptide')
    plt.show()
    '''


if __name__ == '__main__':
    main(sys.argv)
