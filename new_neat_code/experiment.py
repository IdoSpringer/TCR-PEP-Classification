import torch
from new_train import convert_data, get_batches, train_model

amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
print(amino_acids)
amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
print(amino_to_ix)

# todo: how to make data files? (the big question)

data = [("CASTEVGGGQNTLYF", "IKAVYNFATCG", 1.0),
        ("CILRAGYQNFYF", "CASGDGNQAPLF", 1.0),
        ("CILRAGYQNFYF", "IKAVYNFATCG", 0.0)]

tcrs = [token[0] for token in data]
peps = [token[1] for token in data]
signs = [token[2] for token in data]

convert_data(tcrs, peps, amino_to_ix)

batches = get_batches(tcrs, peps, signs, batch_size=3)

# model = SiameseLSTMClassifier(10, 10, 'cpu')

'''
for batch in batches:
    padded_tcrs, tcr_lens, padded_peps, pep_lens, batch_signs = batch
    prob = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
    print(prob)
'''

train_model(batches, 'cuda:0')