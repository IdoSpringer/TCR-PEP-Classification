import keras
from keras.layers import Input, Dense, Activation, Reshape, Dropout
from keras.models import Model
import numpy as np
import os
import csv
import random
from keras.utils import multi_gpu_model
from keras.models import load_model
from sklearn.model_selection import train_test_split
import pickle
import random
import shutil
import math
import pandas as pd


# load data by path - all data, directory or one file
def load_data(path):
    all_data = []
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            # print(file)
            with open(os.path.join(directory, file), mode='r') as infile:
                reader = csv.reader(infile)
                data = [row[1] for row in reader]
                all_data += data[1:]
    # a one file full path
    if len(all_data) == 0:
        with open(path, mode='r') as infile:
            reader = csv.reader(infile)
            data = [row[1] for row in reader]
            all_data = data[1:]
    return all_data


# load data by path - n data, directory or one file
def load_n_data(path, p):
    all_data_n = []
    all_list = []
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            print(file)
            with open(os.path.join(directory, file), mode='r') as infile:
                reader = csv.reader(infile)
                data = [row[1] for row in reader]
                data = [i for i in data[1:] if str(i).find('*') == -1 and str(i).find('X') == -1]
                all_list += data
                # sample some from each sample
                sample_n = int(len(data) * p)
                all_data_n += random.sample(data, sample_n)
    # a one file full path
    if len(all_data_n) == 0:
        with open(path, mode='r') as infile:
            reader = csv.reader(infile)
            data = [row[1] for row in reader]
            data = [i for i in data[1:] if str(i).find('*') == -1 and str(i).find('X') == -1]
            all_list += data
            # sample some from each sample
            sample_n = int(len(data) * p)
            all_data_n = random.sample(data, sample_n)
    # check maximal length
    max_length = np.max([len(s) for s in all_list])
    return all_data_n, max_length


# check if a given directory is a type of host
def check_type(t, dir):
    if t in ['flu', 'pfizer'] and t in dir:
        return True
    if t == 'nina' and ('BM' in dir or 'PBL' in dir):
        return True
    return False


# load type data- all lengths, returns all data in one list
def load_n_type(t):
    all_data = []
    for directory, subdirectories, files in os.walk('D_clones_CDR3s'):
        if check_type(t, directory):
            for file in files:
                with open(os.path.join(directory, file), mode='r') as infile:
                    reader = csv.reader(infile)
                    data = [row[1] for row in reader]
                    data = [i for i in data[1:] if str(i).find('*') == -1 and str(i).find('X') == -1]
                    # sample some from each sample
                    sample_n = int(len(data)*0.1)
                    all_data += random.sample(data, sample_n)
    return all_data


# load all of type data- given length n, returns a dictionary of processed data per file
def load_all_samples_all_lengths(my_type, n):
    samples = dict()
    for directory, subdirectories, files in os.walk('D_clones_CDR3s'):
        if check_type(my_type, directory):
            for file in files:
                with open(os.path.join(directory, file), mode='r') as infile:
                    reader = csv.reader(infile)
                    data = [row[1] for row in reader]
                    # X to missing aa at nina's
                    data = [i for i in data[1:] if str(i).find('*') == -1 and str(i).find('X') == -1]
                    samples[file] = data_preprocessing(data, n)
    return samples


# load all of type data- given length n, returns a dictionary of processed data per file
def load_all_samples_all_lengths_from_path(path, n):
    samples = dict()
    all_summary = []
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            with open(os.path.join(directory, file), mode='r') as infile:
                reader = csv.reader(infile)
                # data = [row[1] for row in reader]
                # # N to missing aa at nina's
                # data = [i for i in data[1:] if str(i).find('*') == -1 and str(i).find('N') == -1]
                # for yoram -----
                data = []
                summary = []
                for row in reader:
                    if str(row[1]).find('*') == -1 and str(row[1]).find('N') == -1:
                        summary.append([file, row[0]])
                        data.append(row[1])
                samples[directory.split('/')[1]+'^'+file] = data_preprocessing(data[1:], n)
                # summary = [[file, row[0]] for row in reader if str(row[1]).find('*') == -1 and str(row[1]).find('N') == -1]
                all_summary += summary[1:]

    return samples, all_summary


# cdr3s to one hot vectors
def data_preprocessing(string_set, max_length):
    # one hot vector per amino acid dictionary- wih STOP sequence- !
    aa = ['V', 'I', 'L', 'E', 'Q', 'D', 'N', 'H', 'W', 'F',
          'Y', 'R', 'K', 'S', 'T', 'M', 'A', 'G', 'P', 'C', '!']
    n_aa = len(aa)
    one_hot = {a: [0] * n_aa for a in aa}
    for key in one_hot:
        one_hot[key][aa.index(key)] = 1
    # add zero key for the zero padding
    one_hot['0'] = [0] * n_aa
    # add 1 to the maximum length ( +1 for the ! stop signal)
    max_length += 1
    # generate one-hot long vector for each cdr3
    one_vecs = []
    for cdr3 in string_set:
        # add stop signal in each sequence
        cdr3 = cdr3 + '!'
        my_len = len(cdr3)
        # zero padding in the end of the sequence
        if my_len < max_length:
            add = max_length - my_len
            cdr3 = cdr3 + '0'*add
        # one hot vectors
        v = []
        for c in cdr3:
            v += one_hot[c]
        one_vecs.append(v)
    return one_vecs


tcrs = {'CASSGPGGAETLYF', 'CASTEVGGGQNTLYF', 'CASGDGNQAPLF'}
max_len = max([len(tcr) for tcr in tcrs])
a = data_preprocessing(tcrs, max_len)
print(len(a), a)
print(len(a[0]), a[0])


def hardmax_zero_padding(l):
    n = 21
    l_chunks = [l[i:i + n] for i in range(0, len(l), n)]
    l_new = []
    for chunk in l_chunks:
        new_chunk = list(np.zeros(n, dtype=int))
        # # taking the max only in place where not everything is 0
        # if not all(v == 0 for v in chunk):
        max = np.argmax(chunk)
        if max == 20:
            break
        new_chunk[max] = 1
        l_new += new_chunk
    return l_new


def count_mismatches_zero_padding(a, b):
    n = 21
    a_chunks = [a[i:i + n] for i in range(0, len(a), n)]
    b_chunks = [b[i:i + n] for i in range(0, len(b), n)]
    count_err = 0
    for ind, chunck_a in enumerate(a_chunks):
        ind_a = ''.join(str(x) for x in chunck_a).find('1')
        ind_b = ''.join(str(x) for x in b_chunks[ind]).find('1')
        if ind_a != ind_b:
            count_err += 1
        # early stopping when there are allready 2 mismatches
        if count_err > 2:
            return 3
    return count_err


def calc_accuracy_zero_padding(inputs, y):
    acc = 0
    acc1 = 0
    acc2 = 0
    n = len(inputs)
    for i in range(n):
        hard_max_y = hardmax_zero_padding(y[i])
        real = list(inputs[i])
        # cut the output to be the same length as input
        real = real[:len(hard_max_y)]
        if real == hard_max_y:
            acc += 1
            acc1 += 1
            acc2 += 1
        else:
            # accept 1 mismatch aa
            err = count_mismatches_zero_padding(real, hard_max_y)
            if err == 1:
                acc1 += 1
                acc2 += 1
            else:
                # accept 2 mismatch aa
                if err == 2:
                    acc2 += 1
    print('accuracy: ' + str(acc) + '/' + str(n) + ', ' + str(round((acc / n) * 100, 2)) + '%')
    print('1 mismatch accuracy: ' + str(acc1) + '/' + str(n) + ', ' + str(round((acc1 / n) * 100, 2)) + '%')
    print('2 mismatch accuracy: ' + str(acc2) + '/' + str(n) + ', ' + str(round((acc2 / n) * 100, 2)) + '%')


class AutoEncoder:
    def __init__(self, input_set, encoding_dim=3):
        self.encoding_dim = encoding_dim
        self.x = np.array(input_set)
        self.input_shape = len(input_set[0])
        print(self.x)

    def _encoder(self):
        inputs = Input(shape=self.x[0].shape)
        print(self.x[0].shape)
        encoded1 = Dense(300, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(encoded1)
        encoded2 = Dense(100, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(encoded2)
        encoded3 = Dense(self.encoding_dim, activation='elu')(dropout2)
        model = Model(inputs, encoded3)
        self.encoder = model
        print(model.summary())
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded1 = Dense(100, activation='elu')(inputs)
        dropout1 = Dropout(0.1)(decoded1)
        decoded2 = Dense(300, activation='elu')(dropout1)
        dropout2 = Dropout(0.1)(decoded2)
        decoded3 = Dense(self.input_shape, activation='elu')(dropout2)
        reshape = Reshape((int(self.input_shape/21), 21))(decoded3)
        decoded3 = Dense(21, activation='softmax')(reshape)
        reshape2 = Reshape(self.x[0].shape)(decoded3)
        model = Model(inputs, reshape2)
        print(model.summary())

        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()
        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)

        self.model = model
        return model

    def fit(self, batch_size=10, epochs=300):
        self.model = multi_gpu_model(self.model, gpus=4)
        adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(optimizer=adam, loss='mse', metrics=['mae'])
        log_dir = './log/'
        tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                  write_graph=True, write_images=True)
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                    patience=10, verbose=0, mode='auto')
        self.model.fit(self.x, self.x, validation_split=0.2, verbose=1,
                       epochs=epochs, batch_size=batch_size,
                       callbacks=[tb_callback, es_callback])

    def save(self):
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')
        else:
            self.encoder.save(r'./weights/encoder_weights.h5')
            self.decoder.save(r'./weights/decoder_weights.h5')
            self.model.save(r'./weights/ae_weights.h5')


if __name__ == '__main__':
    exit()

    # data
    # # nina's data
    # root = 'D_clones_CDR3s'
    # # load samples with all lengths
    # data = load_n_type('nina')

    # BM data
    root = 'BM_data_CDR3s'
    # load samples with all lengths
    data, max_len = load_n_data(root, 0.01)

    vecs_data = data_preprocessing(data, max_len)

    # train + test sets
    train_X, test_X, train_y, test_y = train_test_split(vecs_data, vecs_data, test_size=0.2)

    # train model
    ae = AutoEncoder(train_X, encoding_dim=30)
    ae.encoder_decoder()
    ae.fit(batch_size=50, epochs=500)
    ae.save()

    # test model
    encoder = load_model(r'./weights/encoder_weights.h5')
    decoder = load_model(r'./weights/decoder_weights.h5')
    inputs = np.array(test_X)
    input_vec_size = len(inputs[0])
    cdr3_len = input_vec_size/21
    x = encoder.predict(inputs)
    y = decoder.predict(x)

    # save the CDR3 size for the autoencoder
    pickle.dump(cdr3_len, open('cdr3_len.p', "wb"))

    # accuracy
    calc_accuracy_zero_padding(inputs, y)

    # load CDR3 size for the autoencoder
    n = int(pickle.load(open("cdr3_len.p", "rb"))) - 1  # the stop sequence: !

    # predict autoencoder representation
    # a. load all samples cdr3s
    # # nina's data
    # samples = load_all_samples_all_lengths('nina', n)
    # BM data
    samples, all_data = load_all_samples_all_lengths_from_path(root, n)

    # b. predict samples representation using the autoencoder and save predictions
    dir = 'final_autoencoder_projections_Sol/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)  # removes all the content
        os.makedirs(dir)
    encoder = load_model(r'./weights/encoder_weights.h5')
    for s in samples:
        x = encoder.predict(np.array(samples[s]))
        pickle.dump(x, open(dir + s + '.p', "wb"))

