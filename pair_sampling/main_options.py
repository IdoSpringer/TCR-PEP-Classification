from new_train import *


def main(argv):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}

    # Set all parameters and program arguments
    device = argv[3]
    args = {}
    args['train_auc_file'] = argv[4]
    args['test_auc_file'] = argv[5]
    args['siamese'] = bool(argv[1] == 'siamese')
    params = {}
    params['lr'] = 1e-3
    params['wd'] = 0
    params['epochs'] = 200
    params['batch_size'] = 50
    params['lstm_dim'] = 30
    params['emb_dim'] = 10
    params['dropout'] = 0.1
    params['option'] = 0
    # Load data
    pairs_file = 'pairs_data/weizmann_pairs.txt'
    if argv[-1] == 'cancer':
        pairs_file = 'pairs_data/cancer_pairs.txt'
    if argv[-1] == 'shugay':
        pairs_file = 'pairs_data/shugay_pairs.txt'
    train, test = d.load_data(pairs_file)

    # train
    train_tcrs, train_peps, train_signs = get_lists_from_pairs(train)
    convert_data(train_tcrs, train_peps, amino_to_ix)
    train_batches = get_batches(train_tcrs, train_peps, train_signs, params['batch_size'])

    # test
    test_tcrs, test_peps, test_signs = get_lists_from_pairs(test)
    convert_data(test_tcrs, test_peps, amino_to_ix)
    test_batches = get_batches(test_tcrs, test_peps, test_signs, params['batch_size'])

    # Train the model
    model = train_model(train_batches, test_batches, device, args, params)

    # Save trained model
    torch.save({
                'model_state_dict': model.state_dict(),
                'amino_to_ix': amino_to_ix
                }, argv[2])
    pass


def grid(lrs, wds):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}

    # Set all parameters and program arguments
    device = sys.argv[3]
    args = {}
    args['siamese'] = bool(sys.argv[1] == 'siamese')
    params = {}
    params['epochs'] = 500
    params['batch_size'] = 100

    # Load data
    w_file = 'pairs_data/weizmann_pairs.txt'
    c_file = 'pairs_data/cancer_pairs.txt'
    train_w, test_w = d.load_data(w_file)
    train_c, test_c = d.load_data(c_file)
    option = int(sys.argv[2])
    if option == 1:
        # train on other data, test on cancer
        train = train_w + test_w
        test = train_c + test_c
    elif option == 2:
        # train on all data, test on all data
        train = train_w + train_c
        shuffle(train)
        test = test_w + test_c
        shuffle(test)
    elif option == 3:
        # train on cancer data, test on cancer data
        train = train_c
        test = test_c

    # train
    train_tcrs, train_peps, train_signs = get_lists_from_pairs(train)
    convert_data(train_tcrs, train_peps, amino_to_ix)
    train_batches = get_batches(train_tcrs, train_peps, train_signs, params['batch_size'])

    # test
    test_tcrs, test_peps, test_signs = get_lists_from_pairs(test)
    convert_data(test_tcrs, test_peps, amino_to_ix)
    test_batches = get_batches(test_tcrs, test_peps, test_signs, params['batch_size'])

    # Grid csv file
    grid_file = sys.argv[4]
    with open(grid_file, 'a+') as c:
        c = csv.writer(c, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        c.writerow(['model type', 'option', 'learning rate', 'weight decay', 'train auc score', 'test auc score'])

    # Grid run
    for lr in lrs:
        for wd in wds:
            if args['siamese']:
                key = 's'
            else:
                key = 'd'
            args['train_auc_file'] = 'train_auc2_' + key + str(option) + '_lr' + str(lr) + '_wd' + str(wd)
            args['test_auc_file'] = 'test_auc2_' + key + str(option) + '_lr' + str(lr) + '_wd' + str(wd)
            params['lr'] = lr
            params['wd'] = wd
            model = train_model(train_batches, test_batches, device, args, params)
            train_auc = evaluate(model, train_batches, device)
            test_auc = evaluate(model, test_batches, device)
            with open(grid_file, 'a+') as c:
                c = csv.writer(c, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
                c.writerow([key, option, params['lr'], params['wd'], train_auc, test_auc])
    pass


def train_with_cancer(argv, train_file, test_file):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}

    # Set all parameters and program arguments
    device = argv[4]
    args = {}
    args['train_auc_file'] = argv[5]
    args['test_auc_file'] = argv[6]
    # option 2
    try:
        args['test_auc_file_w'] = argv[6]
        args['test_auc_file_c'] = argv[7]
    except IndexError:
        pass
    args['siamese'] = bool(argv[1] == 'siamese')
    params = {}
    params['lr'] = 1e-3
    params['wd'] = 0
    params['emb_dim'] = 10
    params['lstm_dim'] = 10
    params['epochs'] = 1000
    params['batch_size'] = 100

    # Load data
    train_file1, test_file1 = d.load_data(train_file)
    # Do not split (all is train)

    train_file2, test_file2 = d.load_data(test_file)
    # Do not split (all is test)
    option = int(argv[2])
    params['option'] = option
    if option == 1:
        # train on other data, test on cancer
        train = train_file1 + test_file1
        test = train_file2 + test_file2
    elif option == 2:
        # train on all data, test on all data
        train = train_file1 + train_file2
        shuffle(train)
        test_w = test_file1
        shuffle(test_w)
        test_c = test_file2
        shuffle(test_c)
    elif option == 3:
        # train on cancer data, test on cancer data
        train = train_file2
        test = test_file2

    # print(len(train), len(test))

    # train
    train_tcrs, train_peps, train_signs = get_lists_from_pairs(train)
    convert_data(train_tcrs, train_peps, amino_to_ix)
    train_batches = get_batches(train_tcrs, train_peps, train_signs, params['batch_size'])

    if option == 2:
        # test w
        test_tcrs_w, test_peps_w, test_signs_w = get_lists_from_pairs(test_w)
        convert_data(test_tcrs_w, test_peps_w, amino_to_ix)
        test_batches_w = get_batches(test_tcrs_w, test_peps_w, test_signs_w, params['batch_size'])
        # test c
        test_tcrs_c, test_peps_c, test_signs_c = get_lists_from_pairs(test_c)
        convert_data(test_tcrs_c, test_peps_c, amino_to_ix)
        test_batches_c = get_batches(test_tcrs_c, test_peps_c, test_signs_c, params['batch_size'])
        test_batches = (test_batches_w, test_batches_c)
        pass
    else:
        # test
        test_tcrs, test_peps, test_signs = get_lists_from_pairs(test)
        convert_data(test_tcrs, test_peps, amino_to_ix)
        test_batches = get_batches(test_tcrs, test_peps, test_signs, params['batch_size'])

    # Train the model
    model = train_model(train_batches, test_batches, device, args, params)

    # Save trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'amino_to_ix': amino_to_ix
    }, argv[3])
    pass


def w_grid(lrs, wds, emb_dims, lstm_dims):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}

    # Set all parameters and program arguments
    device = sys.argv[2]
    args = {}
    args['siamese'] = bool(sys.argv[1] == 'siamese')
    params = {}
    params['epochs'] = 600
    params['batch_size'] = 100

    # Load data
    w_file = 'pairs_data/weizmann_pairs.txt'
    train, test = d.load_data(w_file)

    # train
    train_tcrs, train_peps, train_signs = get_lists_from_pairs(train)
    convert_data(train_tcrs, train_peps, amino_to_ix)
    train_batches = get_batches(train_tcrs, train_peps, train_signs, params['batch_size'])

    # test
    test_tcrs, test_peps, test_signs = get_lists_from_pairs(test)
    convert_data(test_tcrs, test_peps, amino_to_ix)
    test_batches = get_batches(test_tcrs, test_peps, test_signs, params['batch_size'])

    # Grid csv file
    grid_file = sys.argv[3]
    with open(grid_file, 'a+') as c:
        c = csv.writer(c, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        c.writerow(['model type', 'embedding dimension', 'lstm dimension', 'learning rate', 'weight decay',
                    'train auc score', 'test auc score'])

    # Grid run
    for emb_dim in emb_dims:
        for lstm_dim in lstm_dims:
            for lr in lrs:
                for wd in wds:
                    if args['siamese']:
                        key = 's'
                    else:
                        key = 'd'
                    args['train_auc_file'] = 'weizmann_grid_auc/w_train_auc_' + key + '_ed' + str(emb_dim)\
                                             + '_od' + str(lstm_dim) + '_lr' + str(lr) + '_wd' + str(wd)
                    args['test_auc_file'] = 'weizmann_grid_auc/w_test_auc_' + key + '_ed' + str(emb_dim)\
                                            + '_od' + str(lstm_dim) + '_lr' + str(lr) + '_wd' + str(wd)
                    params['emb_dim'] = emb_dim
                    params['lstm_dim'] = lstm_dim
                    params['lr'] = lr
                    params['wd'] = wd
                    model = train_model(train_batches, test_batches, device, args, params)
                    train_auc = evaluate(model, train_batches, device)
                    test_auc = evaluate(model, test_batches, device)
                    with open(grid_file, 'a+') as c:
                        c = csv.writer(c, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
                        c.writerow([key, params['emb_dim'], params['lstm_dim'], params['lr'], params['wd'],
                                    train_auc, test_auc])


if __name__ == '__main__':
    main(sys.argv)
    # grid(lrs=[1e-4, 1e-3, 1e-2, 1e-1], wds=[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    # todo solve problem in option 3 (regularization?)
    # todo dimensions grid for weizmann (7, 10)
    # todo add dropout
    # todo more epochs
    # todo separate test evals for weizmann and cancer in option 2
    # train_with_cancer(sys.argv, 'pairs_data/weizmann_pairs.txt', 'pairs_data/cancer_pairs.txt')
    # train_with_cancer(sys.argv, 'pairs_data/weizmann_pairs.txt', 'pairs_data/cancer_10tcr.txt')
    # w_grid(lrs=[1e-3, 1e-2], wds=[0, 1e-5, 1e-6, 1e-4], emb_dims=[7, 10], lstm_dims=[3, 5, 7, 10])

# run:
#   source activate tf_gpu
#   nohup python new_train.py siamese model.pt cuda:2 train_auc test_auc

# grid
# source activate tf_gpu
# nohup python new_train.py cuda:2 grid2_w_lr_wd.csv (siamese do far)

# weizmann grid
# source activate tf_gpu
# nohup python main_options.py siamese cuda:1 weizmann_dim_grid.csv
# nohup python main_options.py double cuda:2 weizmann_dim_d_grid.csv
