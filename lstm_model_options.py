import sys
from pair_sampling.new_train import *
import load_with_tcrgp as d2
import load_netTCR_data as d3
import pickle


def main(argv):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}

    # Set all parameters and program arguments
    device = argv[2]
    args = {}
    args['train_auc_file'] = argv[3]
    args['test_auc_file'] = argv[4]
    args['roc_file'] = argv[5]
    args['siamese'] = False
    params = {}
    params['lr'] = 1e-3
    params['wd'] = 1e-5
    params['epochs'] = 200
    params['batch_size'] = 50
    params['lstm_dim'] = 30
    params['emb_dim'] = 10
    params['dropout'] = 0.1
    params['option'] = 0
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

    train, test = d.load_data(pairs_file)
    if argv[6] == 'tcrgp':
        train, test = d2.load_data(pairs_file)
    if argv[6] == 'nettcr':
        pairs_file = 'netTCR/parameters/iedb_mira_pos_uniq.txt'
        train, test = d3.load_data(pairs_file)

    with open(argv[7] + '.pickle', 'wb') as handle:
        pickle.dump(test, handle)

    # train
    train_tcrs, train_peps, train_signs = get_lists_from_pairs(train)
    convert_data(train_tcrs, train_peps, amino_to_ix)
    train_batches = get_batches(train_tcrs, train_peps, train_signs, params['batch_size'])

    # test
    test_tcrs, test_peps, test_signs = get_lists_from_pairs(test)
    convert_data(test_tcrs, test_peps, amino_to_ix)
    test_batches = get_batches(test_tcrs, test_peps, test_signs, params['batch_size'])

    # Train the model
    model, best_auc, best_roc = train_model(train_batches, test_batches, device, args, params)

    # Save trained model
    torch.save({
                'model_state_dict': model.state_dict(),
                'amino_to_ix': amino_to_ix
                }, argv[1])
    # Save best ROC curve and AUC
    np.savez(args['roc_file'], fpr=best_roc[0], tpr=best_roc[1], auc=np.array(best_auc))
    pass


def pep_test():
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    with open(sys.argv[1], 'rb') as handle:
        test = pickle.load(handle)
    # test
    test_tcrs, test_peps, test_signs = get_lists_from_pairs(test)
    device = 'cuda:3'
    model = DoubleLSTMClassifier(10, 30, 0.1, device)
    checkpoint = torch.load(sys.argv[2])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    """
    McPAS most frequent peps
    LPRRSGAAGA 2145 Influenza
    GILGFVFTL 1598 Influenza
    GLCTLVAML 1071 Epstein Barr virus (EBV)	
    NLVPMVATV 809 Cytomegalovirus (CMV)	
    SSYRRPVGI 653 Influenza
    """
    for pep in ['LPRRSGAAGA', 'GILGFVFTL', 'GLCTLVAML', 'NLVPMVATV', 'SSYRRPVGI']:
        pep_shows = [i for i in range(len(test_peps)) if pep == test_peps[i]]
        test_tcrs_pep = [test_tcrs[i] for i in pep_shows]
        test_peps_pep = [test_peps[i] for i in pep_shows]
        test_signs_pep = [test_signs[i] for i in pep_shows]
        convert_data(test_tcrs_pep, test_peps_pep, amino_to_ix)
        test_batches_pep = get_batches(test_tcrs_pep, test_peps_pep, test_signs_pep, 50)
        if len(pep_shows):
            test_auc, roc = evaluate(model, test_batches_pep, device)
            print(pep, test_auc)
    """
    VDJDB most frequent peps
    NLVPMVATV 4731 Cytomegalovirus (CMV)	
    GILGFVFTL 3132 Influenza
    ELAGIGILTV 1808 Melanoma
    GLCTLVAML 1122 Epstein Barr virus (EBV)	
    TTPESANL 858 Simian immunodeficiency viruses (SIV)
    """


if __name__ == '__main__':
    if len(sys.argv) > 3:
        main(sys.argv)
    else:
        pep_test()
