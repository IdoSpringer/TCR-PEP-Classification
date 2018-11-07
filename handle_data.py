import pickle
import torch


# Load data from files
def load_data():
    """
        Open the data files
    """
    # Open pickle files
    with open('train sequences.pickle', 'rb') as file_1,\
         open('test sequences.pickle', 'rb') as file_2,\
         open('triplets before.pickle', 'rb') as file_3,\
         open('triplets after.pickle', 'rb') as file_4:
        train_lst = pickle.load(file_1)
        test_lst = pickle.load(file_2)
        dict_before = pickle.load(file_3)
        dict_after = pickle.load(file_4)
    return train_lst, test_lst, dict_before, dict_after


# Get x, y representation of data
def data_generator(data_train, data_test, peptides):
    """
        Generate data as x and y lists
    """
    seq_lists = [data_train[pep_name] for pep_name in peptides]
    # x is the word
    x_tr = [x for l in seq_lists for x in l]
    # y is the peptide number
    y_tr = [[i] * len(x) for i, x in enumerate(seq_lists)]
    y_tr = [item for sublist in y_tr for item in sublist]

    seq_lists = [data_test[pep_name] for pep_name in peptides]
    x_te = [x for l in seq_lists for x in l]
    y_te = [[i] * len(x) for i, x in enumerate(seq_lists)]
    y_te = [item for sublist in y_te for item in sublist]

    return x_tr, y_tr, x_te, y_te


# Get letter list and letter-index dicts
def get_letters_seq(data):
    """
        Get letters list and letter-index dictionaries
    """
    # Letters of all words
    a1 = [set(x) for l in data.values() for x in l]
    # All letters (20)
    letter = ['<PAD>'] + list(set([x for l in a1 for x in l]))
    # Letters-index dictionaries
    letter_to_ix = dict((i, j) for j, i in enumerate(letter))
    ix_to_letter = dict((j, i) for j, i in enumerate(letter))
    return letter, letter_to_ix, ix_to_letter


# UNNECESSARY - Same as above?
# Get letter list and letter-index dicts
def get_letters_pep(peptides_lst):
    """
        Get letters list and letter-index dictionaries
    """
    word_to_ix_ = dict((i, j) for j, i in enumerate(['<PAD>']+list(set(x for l in peptides_lst for x in l))))
    ix_to_word_ = dict((j, i) for j, i in enumerate(['<PAD>']+list(set(x for l in peptides_lst for x in l))))
    return word_to_ix_, ix_to_word_


# Yield successive n-sized chunks from data
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    seq_length = np.asarray([len(x[0]) for x in l])

    tmp_l = [[] for i in range(len(set(seq_length)))]
    for i, j in zip(set(seq_length), tmp_l):
        j.extend(np.asarray(l)[seq_length == i])
    for g in tmp_l:
        for i in range(0, len(g), n):
            yield g[i:i + n]


def get_batch(dat, to_ix):
    vectorized_seqs = [[to_ix[l] for l in pep] for pep in dat]
    # get sequences lengths
    seq_lengths = torch.LongTensor([len(x) for x in vectorized_seqs])

    # dump padding everywhere, and place seqs on the left.
    # NOTE: you only need a tensor as big as your longest sequence
    seq_tensor = autograd.Variable(torch.zeros((len(vectorized_seqs), seq_lengths.max()))).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

    # SORTING sequences by length
    seq_lengths, length_id = seq_lengths.sort(0, descending=True)
    # ordering batch
    seq_tensor = seq_tensor[length_id]

    seq_tensor = seq_tensor.transpose(0, 1)  # (B,L,D) -> (L,B,D)

    return seq_tensor, seq_lengths
