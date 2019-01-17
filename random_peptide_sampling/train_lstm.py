import torch.optim as optim
from random_peptide_sampling.models import *
import random
import torch.autograd as autograd
import numpy as np
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
from random_peptide_sampling import handle_data as hd


# todo: added weighted loss by p_vec. check if it is helpful

def train(model, current_data, aux_data, optimizer, loss_function, epoch_pep, device, p_vec):
    model.train()
    word_to_ix, peptides_list, pep_to_ix = aux_data
    total_loss = torch.Tensor([0])
    if device.type != 'cpu':
        total_loss = total_loss.to(device)
    for batch_ in current_data:
        optimizer.zero_grad()

        x_train, y_train = zip(*batch_)
        input_seq, sequences_len = hd.get_batch(x_train, word_to_ix)
        if device.type != 'cpu':
            input_seq = input_seq.to(device)
            sequences_len = sequences_len.to(device)
        lst_of_pep_ix = [epoch_pep] * len(y_train)
        lst_of_pep = [peptides_list[i] for i in lst_of_pep_ix]
        input_pep, peptides_len = hd.get_batch(lst_of_pep, pep_to_ix)
        if device.type != 'cpu':
            input_pep = input_pep.to(device)
            peptides_len = peptides_len.to(device)
        y_predict = model.forward(input_seq, sequences_len, input_pep, peptides_len)
        new_y = (np.asarray(lst_of_pep_ix) == np.asarray(y_train).astype(int)).astype(int)
        target = autograd.Variable(torch.FloatTensor([new_y])).view(-1)
        if device.type != 'cpu':
            target = target.to(device)

        loss = loss_function(y_predict.view(-1), target) * ((1 / p_vec[epoch_pep]) ** 2)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss


def do_one_train(model_name, peptides_lst, data, device, params=None):
    st = time.time()
    # Read arguments
    divide = params['divide']
    n_layers = params['num_layers']
    bi_lstm = params['bi_lstm']
    embedding_dim = params['embedding_dim']
    hidden_dim = params['hidden_dim']

    # Get data
    train_lst, test_lst, letter_to_ix, ix_to_letter, letters = data
    num_of_peptides = len(peptides_lst)
    # Get x,y data representation
    x_train, y_train, x_test, y_test = hd.data_generator(train_lst, test_lst, peptides_lst)
    data = list(zip(x_train, y_train))
    if divide:
        x_dev, x_test, y_dev, y_test = train_test_split(x_test, y_test, test_size=0.3, random_state=42)
    else:
        x_dev, y_dev = x_test, y_test
    pep_to_ix, ix_to_pep = hd.get_letters_pep(peptides_lst)
    aux_data = [letter_to_ix, peptides_lst, pep_to_ix]

    # Building model
    if model_name == 'one':
        model = LSTMClassifierSimple(embedding_dim, hidden_dim, len(letter_to_ix), 1, len(pep_to_ix), device)
    elif model_name == 'double':
        model = DoubleLSTMClassifier(embedding_dim, hidden_dim, len(letter_to_ix), 1, len(pep_to_ix), device)
    elif model_name == 'upgrade':
        model = DoubleLSTMClassifierUpgrade(embedding_dim, hidden_dim, len(letter_to_ix), 1, len(pep_to_ix), device, n_layers, bi_lstm)
    if device.type != 'cpu':
        model = model.to(device)

    # Loss and optimization
    loss_function = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])

    # Results
    lst_result_train = []
    lst_result_dev = []
    lst_result_test = []

    p_vec = np.array([len(train_lst[x]) for x in peptides_lst]) / sum([len(train_lst[x]) for x in peptides_lst])
    stop_early = 0
    # Training
    losses = []
    for epoch in range(100 * num_of_peptides):  # params should be in params file
        # shuffling and divide data
        random.shuffle(data)
        data_divided = hd.chunks(data, 10)  # param file ?
        specific_batch = list(data_divided)
        # choose peptide
        current_pep = np.random.choice(num_of_peptides, 1, replace=False, p=p_vec)[0]
        # Train epoch and get loss
        lss_ = train(model, specific_batch, aux_data, opt, loss_function, current_pep, device, p_vec)
        if 100 * num_of_peptides - epoch <= 20:
            losses.append(round(lss_.item() / len(specific_batch), 5))

        '''
        # save loss for graphics
        with open(params['loss_file'], 'a+') as file:
            file.write("epoch: " + str(epoch) + " loss: "+str(round(lss_.item() / len(specific_batch), 4))+'\n')
        '''

        # print('num of labels: ', num_of_peptides, round(lss_.item() / len(specific_batch), 4))
        if epoch % 50 == 49:
            lst_result_train.append(
                epoch_measures(x_train, y_train, aux_data, model, True, num_of_peptides, device, p_vec))
            lst_result_dev.append(
                epoch_measures(x_dev, y_dev, aux_data, model, True, num_of_peptides, device, p_vec))
            if divide:
                lst_result_test.append(
                    epoch_measures(x_test, y_test, aux_data, model, True, num_of_peptides, device, p_vec))
        # print("do one train lst_results_dev_: ", lst_result_dev)

    # Test confusion matrix
    conf_matrix = confusion_matrix(x_test, y_test, aux_data, num_of_peptides, model, device)
    # print('confusion matrix:')
    # print(conf_matrix)
    with open(params['confusion_matrix_file'], 'a+') as file:
        file.write('confusion matrix:' + '\n')
        file.write(str(conf_matrix) + '\n')

    with open(params['time_file'], 'a+') as file:
        file.write('pep:'+str(num_of_peptides)+', ed:'+str(embedding_dim)+', lstmd:'+str(hidden_dim)+', lr:'+str(params['lr'])+', wd:'+str(params['wd']))
        file.write(", train time: " + str(time.time() - st) + '\n')

    # print(losses)
    loss_mean = np.mean(losses)
    loss_var = np.var(losses)
    # print(loss_mean, loss_var)
    # Print best results
    roc_auc_t, precision_t, recall_t, f1_t, _ = best_results(lst_result_train)
    train_line = print_line(roc_auc_t, precision_t, recall_t, f1_t)
    roc_auc_, precision_, recall_, f1_, max_ind = best_results(lst_result_dev)
    dev_line = print_line(roc_auc_, precision_, recall_, f1_)
    if divide:
        roc_auc, precision, recall, f1 = np.round(lst_result_test[max_ind][0], 5)
        test_line = print_line(roc_auc, precision, recall, f1)
        roc_auc_, precision_, recall_, f1_, max_ind = best_results(lst_result_test)
        test_line_best = print_line(roc_auc_, precision_, recall_, f1_)

        return model, train_line, dev_line, test_line_best, test_line, loss_mean, loss_var
    else:
        return model, train_line, dev_line, loss_mean, loss_var


def best_results(lst_):
    max_ind = np.argmax(np.asarray(lst_).reshape((len(lst_), 4))[:, 0])
    roc_auc, precision, recall, f1 = np.round(lst_[max_ind][0], 5)

    return roc_auc, precision, recall, f1, max_ind


def print_line(roc_auc_, precision_, recall_, f1_):
    return str(roc_auc_) + ',' + str(precision_) + ',' + str(recall_) + ',' + str(f1_)


def evaluation_model(x_data, y_data, aux_data, model_, type_eval, num_of_lbl, device, p_vec):
    model_.eval()
    y_hat = []
    y_hat_rounded = []
    y_true = []
    # y_pred_auc=[]
    word_to_ix, peptides_list, pep_to_ix = aux_data
    data_test = list(zip(x_data, y_data))
    # print("x data: ", x_data, "y data: ", y_data)
    data_divided_test = hd.chunks(data_test, 10)
    specific_batch_test = list(data_divided_test)

    # Initialize confusion matrix
    # confusion_matrix = np.zeros([num_of_lbl, num_of_lbl])

    for batch_test in specific_batch_test:
        x, y = zip(*batch_test)
        # print("x data: ", x, "y data: ", y)
        input_seq, sequences_len = hd.get_batch(x, word_to_ix)
        # print("input seq: ", input_seq, "seq len: ", sequences_len)
        if device.type != 'cpu':
            input_seq = input_seq.to(device)
            sequences_len = sequences_len.to(device)

        current_pep_ = np.random.choice(num_of_lbl, 1, replace=False, p=p_vec)[0]
        # print(current_pep_)
        lst_of_pep_ix = [current_pep_] * len(y)
        lst_of_pep = [peptides_list[i] for i in lst_of_pep_ix]
        input_pep, peptides_len = hd.get_batch(lst_of_pep, pep_to_ix)
        # print("input pep: ", input_pep, "pep len: ", peptides_len)
        if device.type != 'cpu':
            input_pep = input_pep.to(device)
            peptides_len = peptides_len.to(device)

        model_.zero_grad()
        # opt.zero_grad()

        y_predict_ = model_.forward(input_seq, sequences_len, input_pep, peptides_len)
        y_true.extend((np.asarray(lst_of_pep_ix) == np.asarray(y).astype(int)).astype(int))
        # print("y predict: ", y_predict_)
        # y_pred_auc.append(y_predict_.data[0])
        y_hat.extend(y_predict_.view(-1).cpu().data.numpy())
        y_hat_rounded.extend(y_predict_.view(-1).cpu().data.numpy().round())
    # print("y true: ", y_true)
    # print("y hat: ", y_hat)
    # print("y hat rounded: ", y_hat_rounded)
    precision, recall, fbeta_score, _ = metrics.precision_recall_fscore_support(y_true, y_hat_rounded, average='binary')
    # An evaluation for a continuous binary classifier (because rounding the sigmoid is bad)
    roc_auc = metrics.roc_auc_score(y_true, y_hat)
    # print(roc_auc)
    # why it is 0 often?
    '''
    print("y true: ", y_true)
    print("y hat: ", y_hat)
    print("y hat rounded: ", y_hat_rounded)
    print(precision, recall, fbeta_score)
    print(roc_auc)
    assert precision != 0.0
    assert recall != 0.0
    assert fbeta_score != 0.0
    '''
    return precision, recall, fbeta_score, roc_auc


def epoch_measures(x_dat, y_dat, aux_data, model, type_e, num_of_peptides, device, P):
    lst_result_ = []
    precision_lst, recall_lst, fbeta_lst, roc_auc_lst = zip(
        *[evaluation_model(x_dat, y_dat, aux_data, model, type_e, num_of_peptides, device, P)
          for i in range(10)])
    # max_ind, min_ind = np.argmax(fbeta_lst), np.argmin(fbeta_lst)
    max_ind = np.argmax(roc_auc_lst)
    lst_result_.append((roc_auc_lst[max_ind], precision_lst[max_ind], recall_lst[max_ind], fbeta_lst[max_ind]))
    # print("epoch measures lst_results_: ", lst_result_)
    return lst_result_


# todo: confusion matrix
def confusion_matrix(x_data, y_data, aux_data, num_of_labels, model_, device):
    model_.eval()
    word_to_ix, peptides_list, pep_to_ix = aux_data
    data_test = list(zip(x_data, y_data))
    data_divided_test = hd.chunks(data_test, 10)
    specific_batch_test = list(data_divided_test)
    # Initialize confusion matrix
    conf_matrix = np.zeros([num_of_labels, num_of_labels])
    for batch_test in specific_batch_test:
        x, y = zip(*batch_test)
        # print("x data: ", x, "y data: ", y)
        input_seq, sequences_len = hd.get_batch(x, word_to_ix)
        # print("input seq: ", input_seq, "seq len: ", sequences_len)
        if device.type != 'cpu':
            input_seq = input_seq.to(device)
            sequences_len = sequences_len.to(device)
        label_predictions = torch.zeros([len(y), num_of_labels])
        # print('label predictions:', label_predictions)
        # Get prediction for every peptide separately (to make it multiclass prediction)
        for label in range(num_of_labels):
            current_pep_ = label
            # current_pep_ = np.random.choice(num_of_lbl, 1, replace=False, p=p_vec)[0]
            # print('label:', current_pep_)
            lst_of_pep_ix = [current_pep_] * len(y)
            lst_of_pep = [peptides_list[i] for i in lst_of_pep_ix]
            input_pep, peptides_len = hd.get_batch(lst_of_pep, pep_to_ix)
            # print("input pep: ", input_pep, "pep len: ", peptides_len)
            if device.type != 'cpu':
                input_pep = input_pep.to(device)
                peptides_len = peptides_len.to(device)
            model_.zero_grad()
            # opt.zero_grad()
            y_predict_ = model_.forward(input_seq, sequences_len, input_pep, peptides_len)
            # print("y predict: ", y_predict_)
            label_predictions[:, label] = y_predict_.view(-1)
        # print('label predictions:', label_predictions)
        # Take the maximum - multiclass prediction
        prediction = np.asarray(torch.argmax(label_predictions, dim=1)).astype(int)
        # print('label prediction:', prediction)
        y_true = np.asarray(y).astype(int)
        # print('true:', y_true)
        # Update confusion matrix for every sample
        for true_label, predicted_label in zip(y_true, prediction):
            # print(true_label, predicted_label)
            conf_matrix[true_label][predicted_label] += 1
    # print('confusion matrix:', conf_matrix)
    return conf_matrix
