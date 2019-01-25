import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from random import shuffle
import time
import numpy as np
import torch.autograd as autograd
from new_models import SiameseLSTMClassifier, DoubleLSTMClassifier
import load_data as d
from sklearn.metrics import roc_auc_score, roc_curve
import csv


def get_lists_from_pairs(pairs):
    tcrs = []
    peps = []
    signs = []
    for pair in pairs:
        tcr, pep, label, weight = pair
        tcrs.append(tcr)
        peps.append(pep)
        if label == 'p':
            signs.append(1.0)
        elif label == 'n':
            signs.append(0.0)
    return tcrs, peps, signs


def convert_data(tcrs, peps, amino_to_ix):
    for i in range(len(tcrs)):
        if any(letter.islower() for letter in tcrs[i]):
            print(tcrs[i])
        tcrs[i] = [amino_to_ix[amino] for amino in tcrs[i]]
    for i in range(len(peps)):
        peps[i] = [amino_to_ix[amino] for amino in peps[i]]


def get_batches(tcrs, peps, signs, batch_size):
    """
    Get batches from the data
    """
    # Initialization
    batches = []
    index = 0
    # Go over all data
    while index < len(tcrs):
        # Get batch sequences and math tags
        batch_tcrs = tcrs[index:index + batch_size]
        batch_peps = peps[index:index + batch_size]
        batch_signs = signs[index:index + batch_size]
        # Update index
        index += batch_size
        # Pad the batch sequences
        padded_tcrs, tcr_lens = pad_batch(batch_tcrs)
        padded_peps, pep_lens = pad_batch(batch_peps)
        # Add batch to list
        batches.append((padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs))
    # Return list of all batches
    return batches


def pad_batch(seqs):
    """
    Pad a batch of sequences (part of the way to use RNN batching in PyTorch)
    """
    # Tensor of sequences lengths
    lengths = torch.LongTensor([len(seq) for seq in seqs])
    # The padding index is 0
    # Batch dimensions is number of sequences * maximum sequence length
    longest_seq = max(lengths)
    batch_size = len(seqs)
    # Pad the sequences. Start with zeros and then fill the true sequence
    padded_seqs = autograd.Variable(torch.zeros((batch_size, longest_seq))).long()
    for i, seq_len in enumerate(lengths):
        seq = seqs[i]
        padded_seqs[i, 0:seq_len] = torch.LongTensor(seq[:seq_len])
    # Return padded batch and the true lengths
    return padded_seqs, lengths


def train_epoch(batches, model, loss_function, optimizer, device):
    model.train()
    shuffle(batches)
    total_loss = 0
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        batch_signs = torch.tensor(batch_signs).to(device)
        model.zero_grad()
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        # print(probs, batch_signs)
        # Compute loss
        loss = loss_function(probs, batch_signs)
        # with open(sys.argv[1], 'a+') as loss_file:
        #    loss_file.write(str(loss.item()) + '\n')
        # Update model weights
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # print('current loss:', loss.item())
        # print(probs, batch_signs)
    # Return average loss
    return total_loss / len(batches)


def train_model(batches, test_batches, device, args, params):
    """
    Train and evaluate the model
    """
    losses = []
    # We use Cross-Entropy loss
    loss_function = nn.BCELoss()
    # Set model with relevant parameters
    if args['siamese'] is True:
        model = SiameseLSTMClassifier(10, 10, device)  # todo
    else:
        model = DoubleLSTMClassifier(10, 10, device)
    # Move to GPU
    model.to(device)
    # We use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    # Train several epochs
    for epoch in range(params['epochs']):
        print('epoch:', epoch + 1)
        epoch_time = time.time()
        # Train model and get loss
        loss = train_epoch(batches, model, loss_function, optimizer, device)
        losses.append(loss)
        # Compute auc
        train_auc = evaluate(model, batches, device)
        print('train auc:', train_auc)
        with open(args['train_auc_file'], 'a+') as file:
            file.write(str(train_auc) + '\n')
        test_auc = evaluate(model, test_batches, device)
        print('test auc:', test_auc)
        with open(args['test_auc_file'], 'a+') as file:
            file.write(str(test_auc) + '\n')
        print('one epoch time:', time.time() - epoch_time)
    # Print train losses
    # print(losses)
    return model


def evaluate(model, batches, device):
    model.eval()
    true = []
    scores = []
    shuffle(batches)
    for batch in batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        # print(np.array(batch_signs).astype(int))
        # print(probs.cpu().data.numpy())
        true.extend(np.array(batch_signs).astype(int))
        scores.extend(probs.cpu().data.numpy())
    # Return auc score
    auc = roc_auc_score(true, scores)
    # print('auc:', auc)
    return auc


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
    params['epochs'] = 500
    params['batch_size'] = 100

    # Load data
    pairs_file = 'pairs_data/weizmann_pairs.txt'
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


def eval_with_cancer(argv, train_file, test_file):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}

    # Set all parameters and program arguments
    device = argv[4]
    args = {}
    args['train_auc_file'] = argv[5]
    args['test_auc_file'] = argv[6]
    args['siamese'] = bool(argv[1] == 'siamese')
    params = {}
    params['lr'] = 1e-3
    params['wd'] = 0
    params['epochs'] = 10
    params['batch_size'] = 100

    # Load data
    train_file1, test_file1 = d.load_data(train_file)
    # Do not split (all is train)

    train_file2, test_file2 = d.load_data(test_file)
    # Do not split (all is test)
    option = int(argv[2])
    if option == 1:
        # train on other data, test on cancer
        train = train_file1 + test_file1
        test = train_file2 + test_file2
    elif option == 2:
        # train on all data, test on all data
        train = train_file1 + train_file2
        shuffle(train)
        test = test_file1 + test_file2
        shuffle(test)
    elif option == 3:
        # train on cancer data, test on cancer data
        train = train_file2
        test = test_file2

    print(len(train), len(test))

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
    }, argv[3])
    pass


if __name__ == '__main__':
    # main(sys.argv)
    grid(lrs=[1e-4, 1e-3, 1e-2, 1e-1], wds=[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    # todo plots/csv for cancer_grid1 - report
    # todo solve problem in option 3 (regularization?)
    # todo dimensions grid
    # eval_with_cancer(sys.argv, 'pairs_data/weizmann_pairs.txt', 'pairs_data/cancer_pairs.txt')

# run:
#   source activate tf_gpu
#   nohup python new_train.py siamese model.pt cuda:2 train_auc test_auc

# grid
# source activate tf_gpu
# nohup python new_train.py cuda:2 grid2_w_lr_wd.csv (siamese do far)
