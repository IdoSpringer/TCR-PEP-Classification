import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from random import shuffle
import time
import numpy as np
import torch.autograd as autograd
from new_models import SiameseLSTMClassifier
import load_data as d
from sklearn.metrics import roc_auc_score, roc_curve


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
        with open(sys.argv[1], 'a+') as loss_file:
            loss_file.write(str(loss.item()) + '\n')
        # Update model weights
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # print('current loss:', loss.item())
        # print(probs, batch_signs)
    # Return average loss
    return total_loss / len(batches)


def train_model(batches, test_batches, device):
    """
    Train and evaluate the model
    """
    losses = []
    # We use Cross-Entropy loss
    loss_function = nn.BCELoss()
    # Set model with relevant parameters
    model = SiameseLSTMClassifier(10, 10, device)
    # Move to GPU
    model.to(device)
    # We use Adam optimizer
    optimizer = optim.Adam(model.parameters())
    # Train several epochs
    for epoch in range(50):
        print('epoch:', epoch + 1)
        epoch_time = time.time()
        # Train model and get loss
        loss = train_epoch(batches, model, loss_function, optimizer, device)
        losses.append(loss)
        # Compute auc
        train_auc = evaluate(model, batches, device)
        print('train auc:', train_auc)
        with open(sys.argv[4], 'a+') as file:
            file.write(str(train_auc) + '\n')
        test_auc = evaluate(model, test_batches, device)
        print('test auc:', test_auc)
        with open(sys.argv[5], 'a+') as file:
            file.write(str(test_auc) + '\n')
        # print('one epoch time:', time.time() - epoch_time)
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
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    # print(amino_acids)
    amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    # print(amino_to_ix)

    pairs_file = 'pairs_data/weizmann_pairs.txt'
    train, test = d.load_data(pairs_file)
    # print(len(train), train)
    # print(len(test), test)

    train_tcrs, train_peps, train_signs = get_lists_from_pairs(train)
    convert_data(train_tcrs, train_peps, amino_to_ix)
    train_batches = get_batches(train_tcrs, train_peps, train_signs, batch_size=10)

    test_tcrs, test_peps, test_signs = get_lists_from_pairs(test)
    convert_data(test_tcrs, test_peps, amino_to_ix)
    test_batches = get_batches(test_tcrs, test_peps, test_signs, batch_size=10)

    device = argv[3]
    model = train_model(train_batches, test_batches, device)
    torch.save({
                'model_state_dict': model.state_dict(),
                'amino_to_ix': amino_to_ix
                }, argv[2])
    pass


if __name__ == '__main__':
    main(sys.argv)

