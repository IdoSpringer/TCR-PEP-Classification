import cProfile, pstats
from train_lstm import *
from models import *
import handle_data as hd
from handle_params import arguments


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

    print("nohup test")

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
    train_lst, test_lst, dict_before, dict_after = hd.load_data()
    # Get letters list and letter-index dictionaries
    letters, letter_to_ix, ix_to_letter = hd.get_letters_seq(train_lst)
    # Set data
    data = train_lst, test_lst, letter_to_ix, ix_to_letter, letters
    # Sort train by length of lists
    sorted_train_lst = sorted(train_lst, key=lambda k: len(train_lst[k]), reverse=True)

    # Training
    peptides_list = sorted_train_lst[:num_of_peptides]
    best_f1_test = 0.
    best_model = None

    # Grid run
    '''
    for model_name in model_list:
        for embedding_dim in embedding_dims_list:
            for hidden_dim in hidden_dims_list:
                arguments['embedding_dim'] = embedding_dim
                arguments['hidden_dim'] = hidden_dim
                arguments['mode_name'] = model_name
    '''

    for trial in range(1):  # unnecessary?
        if divide:
            fit_model, train_line, dev_line, test_line_best, test_line = do_one_train(model_name,
                                                                                      peptides_list,
                                                                                      data, device,
                                                                                      arguments)
        else:
            fit_model, train_line, dev_line = do_one_train(model_name, peptides_list, data, device,
                                                           arguments)
            test_line = dev_line
        curr_result = float(test_line.split(',')[-2])
        print(curr_result)

        # Write results in file
        with open("save_results.txt", "a+") as my_file:
            my_file.write(fit_model.name_model() + ' ' + str(num_of_peptides) + train_line + 'train' + '\n')
            my_file.write(fit_model.name_model() + ' ' + str(num_of_peptides) + dev_line + 'dev' + '\n')
            if divide:
                my_file.write(
                    fit_model.name_model() + ' ' + str(num_of_peptides) + test_line + 'test' + '\n')
                my_file.write(fit_model.name_model() + ' ' + str(
                    num_of_peptides) + test_line_best + 'test best' + '\n')

        # Best results and model
        if curr_result > best_f1_test:
            best_f1_test = curr_result
            if divide:
                best_model = fit_model, train_line, dev_line, test_line_best, test_line
            else:
                best_model = fit_model, train_line, dev_line

    if divide:
        fit_model, train_line, dev_line, test_line_best, test_line = best_model
    else:
        fit_model, train_line, dev_line = best_model
    print(test_line)

    # Save results to file
    if save_to_file:
        with open(file_name, "a") as my_file:
            my_file.write(fit_model.name_model() + ' ' + str(num_of_peptides) + train_line + 'train' + '\n')
            my_file.write(fit_model.name_model() + ' ' + str(num_of_peptides) + dev_line + 'dev' + '\n')
            if divide:
                my_file.write(
                    fit_model.name_model() + ' ' + str(num_of_peptides) + test_line + 'test' + '\n')
                my_file.write(fit_model.name_model() + ' ' + str(
                    num_of_peptides) + test_line_best + 'test best' + '\n')

    # saving model
    if saving_model:
        torch.save(fit_model, model_file_name + '.pt')
        torch.save(fit_model.state_dict(), model_file_name + '_param.pt')

    # Profiling
    pr.disable()
    pr.dump_stats(arguments['profile_file'])
    p = pstats.Stats(arguments['profile_file'])
    p.sort_stats('cumulative').print_stats(10)


if __name__ == '__main__':
    main()
