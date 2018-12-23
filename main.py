import cProfile, pstats
from train_lstm import *
from models import *
import handle_data as hd
from handle_params import arguments


# TODO: shugay data runs
# TODO: confusion matrix


def main():
    """
    num_lstm: number of lstm (max 2),
    bi_lstm: True for bi-lstm,
    "num_layers": number of lstm layers,
    "dropout": prob for  dropout (at least two layers,
    "save_to_file": True for save results,
    "file_name": name file *.csv,
    "num_of_peptides": number of peptides to run,
    "cuda": gpu device
    """
    # Profiling
    pr = cProfile.Profile()
    pr.enable()


    # Read arguments
    # TODO move to correct files
    cuda = str(arguments['cuda'])
    num_of_training = arguments['num_of_training']
    num_of_peptides = int(arguments['num_of_peptides'])
    model_name = arguments['model_name']
    divide = arguments['divide']
    save_to_file = bool(arguments['save_to_file'])
    file_name = arguments['file_name']
    saving_model = bool(arguments['saving model'])
    model_file_name = arguments['model file name']

    # Grid run chosen parameters
    '''
    num_of_peptides = 2  # PARAM FILE
    embedding_dims_list = [5]  # ? find smart numbers
    hidden_dims_list = [5]
    model_list = ['one']
    # model_list = ['one', 'double', 'upgrade']
    '''

    # GPU
    cuda_device_num = "cuda:" + cuda
    device = torch.device(cuda_device_num if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    gpu = torch.cuda.is_available()
    print(gpu)

    # Load the data from files
    train_lst, test_lst, dict_before, dict_after = hd.load_data(arguments['train_file'], arguments['test_file'])
    # Get letters list and letter-index dictionaries
    letters, letter_to_ix, ix_to_letter = hd.get_letters_seq(train_lst)
    # Set data
    data = train_lst, test_lst, letter_to_ix, ix_to_letter, letters
    # Sort train by length of lists
    sorted_train_lst = sorted(train_lst, key=lambda k: len(train_lst[k]), reverse=True)

    # Training
    peptides_list = sorted_train_lst[:num_of_peptides]
    # best_f1_test = 0.
    # best_model = None

    # Grid run

    # Write results in file
    grid_file = arguments['output_file']
    with open(grid_file, "a+") as file:
        file.write('"Number of peptides","Embedding dimension","LSTM dimension","Learning rate","Weight decay",' +
                   '"Train ROC AUC","Train precision","Train recall","Train F1 score",' +
                   '"Test ROC AUC","Test precision","Test recall","Test F1 score",' +
                   '"Loss mean","Loss variance"' + '\n')
    good_params = []
    # good_params.append({'ed': 10, 'hid': 10, 'lr': 1e-2, 'wd': 1e-6})
    # parameters that give high auc for all peps in weizmann:
    good_params.append({'ed': 10, 'hid': 10, 'lr': 1e-3, 'wd': 1e-6})
    # good_params.append({'ed': 5, 'hid': 10, 'lr': 1e-2, 'wd': 1e-5})
    # good_params.append({'ed': 5, 'hid': 5, 'lr': 1e-2, 'wd': 1e-8})
    # good_params.append({'ed': 10, 'hid': 7, 'lr': 1e-2, 'wd': 1e-7})

    for pep_num in [2,3,4,5,6]:
        for param_dict in good_params:
            num_of_peptides = pep_num
            peptides_list = sorted_train_lst[:num_of_peptides]
            arguments['lr'] = param_dict['lr']
            arguments['wd'] = param_dict['wd']
            arguments['embedding_dim'] = param_dict['ed']
            arguments['hidden_dim'] = param_dict['hid']
            # best_auc_test = 0.
            # best_model = None
            for trial in range(1):
                if divide:
                    fit_model, train_line, dev_line, test_line_best, test_line, loss_mean, loss_var = do_one_train(
                        model_name,
                        peptides_list,
                        data, device,
                        arguments)
                else:
                    fit_model, train_line, dev_line, loss_mean, loss_var = do_one_train(model_name,
                                                                                        peptides_list, data,
                                                                                        device,
                                                                                        arguments)
                    test_line = dev_line
                # roc_auc_score
                # curr_result = float(test_line.split(',')[-5])
                # print(curr_result)

                # Write results in file
                with open(grid_file, "a+") as file:
                    file.write('"' + str(num_of_peptides) + '",')
                    file.write('"' + str(param_dict['ed']) + '",' + '"' + str(param_dict['hid']) + '",' + '"' + str(param_dict['lr']) + '",' + '"' + str(param_dict['wd']) + '",')
                    train_results = train_line.split(',')
                    file.write('"' + str(train_results[0]) + '",' + '"' + str(train_results[1]) + '",' + '"' +str(train_results[2]) + '",' + '"' + str(train_results[3]) + '",')
                    test_results = test_line.split(',')
                    file.write('"' + str(test_results[0]) + '",' + '"' + str(test_results[1]) + '",' + '"' +str(test_results[2]) + '",' + '"' + str(test_results[3]) + '",')
                    file.write('"' + str(loss_mean) + '",' + '"' + str(loss_var) + '"' + '\n')
                print(test_line)
                '''
                # Write results in file
                with open("save_results_grid.txt", "a+") as my_file:
                    my_file.write(fit_model.name_model() + ' ' + str(num_of_peptides) + train_line + 'train' + '\n')
                    my_file.write(fit_model.name_model() + ' ' + str(num_of_peptides) + dev_line + 'dev' + '\n')
                    if divide:
                        my_file.write(
                            fit_model.name_model() + ' ' + str(num_of_peptides) + test_line + 'test' + '\n')
                        my_file.write(fit_model.name_model() + ' ' + str(
                            num_of_peptides) + test_line_best + 'test best' + '\n')
                '''
                '''
                # Best results and model
                if curr_result > best_f1_test:
                    best_auc_test = curr_result
                    if divide:
                        best_model = fit_model, train_line, dev_line, test_line_best, test_line
                    else:
                        best_model = fit_model, train_line, dev_line
                '''
            '''
            if divide:
                fit_model, train_line, dev_line, test_line_best, test_line = best_model
            else:
                fit_model, train_line, dev_line = best_model
            '''
            '''
            # Save results to file
            if save_to_file:
                with open(file_name, "a") as my_file:
                    my_file.write("lr: " + str(lr) + "wd: " + str(wd) + '\n')
                    my_file.write(
                        fit_model.name_model() + ' ' + str(num_of_peptides) + train_line + 'train' + '\n')
                    my_file.write(fit_model.name_model() + ' ' + str(num_of_peptides) + dev_line + 'dev' + '\n')
                    if divide:
                        my_file.write(
                            fit_model.name_model() + ' ' + str(num_of_peptides) + test_line + 'test' + '\n')
                        my_file.write(fit_model.name_model() + ' ' + str(
                            num_of_peptides) + test_line_best + 'test best' + '\n')
            '''
        '''
        for ed in [5]:
            for hid in [10]:
                for wd in [1e-6]:
                    for lr in [1e-3]:
        '''

    '''
    # saving model
    if saving_model:
        torch.save(fit_model, model_file_name + '.pt')
        torch.save(fit_model.state_dict(), model_file_name + '_param.pt')
    '''
    # Profiling
    pr.disable()
    pr.dump_stats(arguments['profile_file'])
    p = pstats.Stats(arguments['profile_file'])
    p.sort_stats('cumulative').print_stats(10)


if __name__ == '__main__':
    main()
