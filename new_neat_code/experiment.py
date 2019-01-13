import torch
import new_train as t
import load_data as d

amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
print(amino_acids)
amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
print(amino_to_ix)

pairs_file = 'pairs_data/weizmann_pairs.txt'
train, test = d.load_data(pairs_file)
print(len(train), train)
print(len(test), test)

tcrs, peps, signs = t.get_lists_from_pairs(train)

t.convert_data(tcrs, peps, amino_to_ix)

batches = t.get_batches(tcrs, peps, signs, batch_size=10)

device = 'cuda:0'
t.train_model(batches, device)