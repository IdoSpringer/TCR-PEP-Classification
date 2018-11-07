import torch.optim as optim
from models import *
import random
import torch.autograd as autograd
import numpy as np
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
import handle_data as hd


def train(model, current_data, aux_data, optimizer, loss_function, epoch_pep, device):
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

        loss = loss_function(y_predict.view(-1), target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss


def do_one_train(model_name, peptides_lst, data, device, params=None):
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
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # Loss and optimization
    loss_function = nn.BCELoss()
    opt = optim.Adam(model.parameters(), weight_decay=1e-3)

    # Results
    lst_result_train = []
    lst_result_dev = []
    lst_result_test = []

    p_vec = np.array([len(train_lst[x]) for x in peptides_lst]) / sum([len(train_lst[x]) for x in peptides_lst])
    ts_ = time.time()
    # Training
    for epoch in range(250 * num_of_peptides):  # params should be in params file
        # shuffling and divide data
        random.shuffle(data)
        data_divided = hd.chunks(data, 10)  # param file ?
        specific_batch = list(data_divided)
        # choose peptide
        current_pep = np.random.choice(num_of_peptides, 1, replace=False, p=p_vec)[0]
        lss_ = train(model, specific_batch, aux_data, opt, loss_function, current_pep, device)

        if epoch % 2 == 0:
            print('num of labels: ', num_of_peptides, round(lss_.item() / len(specific_batch), 4))

            if epoch > 60:
                lst_result_train.append(
                    epoch_measures(x_train, y_train, aux_data, model, True, num_of_peptides, device, p_vec))
                lst_result_dev.append(
                    epoch_measures(x_dev, y_dev, aux_data, model, True, num_of_peptides, device, p_vec))
                if divide:
                    lst_result_test.append(
                        epoch_measures(x_test, y_test, aux_data, model, True, num_of_peptides, device, p_vec))

    print(time.time() - ts_)

    # Print best results
    precision_t, recall_t, f1_t, _ = best_results(lst_result_train)
    train_line = print_line(precision_t, recall_t, f1_t)
    precision_, recall_, f1_, max_ind = best_results(lst_result_dev)
    dev_line = print_line(precision_, recall_, f1_)
    if divide:
        precision_, recall_, f1_, max_ind = best_results(lst_result_test)
        test_line_best = print_line(precision_, recall_, f1_)
        f1, precision, recall = np.round(lst_result_dev[max_ind][0], 5)
        test_line = print_line(precision, recall, f1)

        return model, train_line, dev_line, test_line_best, test_line
    else:
        return model, train_line, dev_line


def best_results(lst_):
    max_ind = np.argmax(np.asarray(lst_).reshape((len(lst_), 3))[:, 0])
    f1, precision, recall = np.round(lst_[max_ind][0], 5)

    return precision, recall, f1, max_ind


def print_line(precision_, recall_, f1_):
    return ',' + str(precision_) + ',' + str(recall_) + ',' + str(f1_) + ','


def evaluation_model(x_data, y_data, aux_data, model_, type_eval, num_of_lbl, device, p_vec):
    model_.eval()
    y_hat = []
    y_true = []
    # y_pred_auc=[]
    word_to_ix, peptides_list, pep_to_ix = aux_data
    data_test = list(zip(x_data, y_data))
    data_divided_test = chunks(data_test, 10)
    specific_batch_test = list(data_divided_test)
    for batch_test in specific_batch_test:
        x, y = zip(*batch_test)
        input_seq, sequences_len = hd.get_batch(x, word_to_ix)
        if device.type != 'cpu':
            input_seq = input_seq.to(device)
            sequences_len = sequences_len.to(device)

        current_pep_ = np.random.choice(num_of_lbl, 1, replace=False, p=p_vec)[0]
        lst_of_pep_ix = [current_pep_] * len(y)
        lst_of_pep = [peptides_list[i] for i in lst_of_pep_ix]
        input_pep, peptides_len = hd.get_batch(lst_of_pep, pep_to_ix)
        if device.type != 'cpu':
            input_pep = input_pep.to(device)
            peptides_len = peptides_len.to(device)

        model_.zero_grad()
        # opt.zero_grad()

        y_predict_ = model_.forward(input_seq, sequences_len, input_pep, peptides_len)
        y_true.extend((np.asarray(lst_of_pep_ix) == np.asarray(y).astype(int)).astype(int))

        # y_pred_auc.append(y_predict_.data[0])
        y_hat.extend(y_predict_.view(-1).cpu().data.numpy().round())

    precision, recall, fbeta_score, _ = metrics.precision_recall_fscore_support(y_true, y_hat, average='binary')

    return precision, recall, fbeta_score


def epoch_measures(x_dat, y_dat, aux_data, model, type_e, num_of_peptides, device, P):
    lst_result_ = []
    precision_lst, recall_lst, fbeta_lst = zip(
        *[evaluation_model(x_dat, y_dat, aux_data, model, type_e, num_of_peptides, device, P)
          for i in range(20)])
    max_ind, min_ind = np.argmax(fbeta_lst), np.argmin(fbeta_lst)
    lst_result_.append((fbeta_lst[max_ind], precision_lst[max_ind], recall_lst[max_ind]))

    return lst_result_

