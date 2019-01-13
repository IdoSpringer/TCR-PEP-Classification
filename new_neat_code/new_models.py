import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SiameseLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, device):
        super(SiameseLSTMClassifier, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        # Embedding matrix - 20 amino acids + padding
        self.embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        # RNN - LSTM
        self.lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=1, batch_first=True)
        # MLP
        self.hidden_layer = nn.Linear(lstm_dim * 2, lstm_dim)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(lstm_dim, 1)
        # self.hidden = self.init_hidden()

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(1, batch_size, self.lstm_dim)).to(self.device),
                autograd.Variable(torch.zeros(1, batch_size, self.lstm_dim)).to(self.device))

    def lstm_pass(self, padded_embeds, lengths):
        # Before using PyTorch pack_padded_sequence we need to order the sequences batch by descending sequence length
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        # Pack the batch and ignore the padding
        padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths, batch_first=True)
        # Initialize the hidden state
        batch_size = len(lengths)
        hidden = self.init_hidden(batch_size)
        # Feed into the RNN
        lstm_out, hidden = self.lstm(padded_embeds, hidden)
        # Unpack the batch after the RNN
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # Remember that our outputs are sorted. We want the original ordering
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        lengths = lengths[unperm_idx]
        return lstm_out

    def forward(self, tcrs, tcr_lens, peps, pep_lens):
        # TCR Encoder:
        # Embedding
        tcr_embeds = self.embedding(tcrs)
        # LSTM Acceptor
        tcr_lstm_out = self.lstm_pass(tcr_embeds, tcr_lens)
        tcr_last_cell = torch.cat([tcr_lstm_out[i, j.data - 1] for i, j in enumerate(tcr_lens)]).view(len(tcr_lens), self.lstm_dim)

        # PEPTIDE Encoder:
        # Embedding
        pep_embeds = self.embedding(peps)
        # LSTM Acceptor
        pep_lstm_out = self.lstm_pass(pep_embeds, pep_lens)
        pep_last_cell = torch.cat([pep_lstm_out[i, j.data - 1] for i, j in enumerate(pep_lens)]).view(len(pep_lens), self.lstm_dim)

        # MLP Classifier
        tcr_pep_concat = torch.cat([tcr_last_cell, pep_last_cell], 1)
        hidden_output = self.relu(self.hidden_layer(tcr_pep_concat))
        mlp_output = self.output_layer(hidden_output)
        output = F.sigmoid(mlp_output)
        return output


# class DoubleLSTMClassifier(nn.Module):
#    pass


class LSTMClassifierSimple(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, device):
        super(LSTMClassifierSimple, self).__init__()
        # GPU
        self.device = device_
        self.hidden_dim = hidden_dim
        # embedding matrix
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_embeddings_2 = nn.Embedding(vocab_size2, embedding_dim, padding_idx=0)
        # LStm
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)
        # The linear layer that connected to dense
        self.hidden1 = nn.Linear(hidden_dim + hidden_dim, hidden_dim // 2)
        self.relu = torch.nn.LeakyReLU()
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim // 2, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, size_batch=1):
        if self.device.type != 'cpu':
            return (autograd.Variable(torch.zeros(1, size_batch, self.hidden_dim)).to(self.device),
                    autograd.Variable(torch.zeros(1, size_batch, self.hidden_dim)).to(self.device))
        else:
            return (autograd.Variable(torch.zeros(1, size_batch, self.hidden_dim)),
                    autograd.Variable(torch.zeros(1, size_batch, self.hidden_dim)))

    def forward(self, inputs, sequences_len, input_pep, peptides_len):
        batch_s = input_pep.size()[-1]
        # PEPTIDE net
        self.hidden = self.init_hidden(batch_s)
        # embed your sequences
        embedding_pep = self.word_embeddings_2(input_pep)
        # pack them up nicely
        packed_input = pack_padded_sequence(embedding_pep, peptides_len)
        # now run through LSTM
        packed_output, (self.hidden, ct) = self.lstm(packed_input, self.hidden)
        # undo the packing operation
        output_pep, length_s = pad_packed_sequence(packed_output)
        output_pep = output_pep.transpose(0, 1)
        # taking the last LSTM cell for each
        pep_to_concat = torch.cat([output_pep[i, j.data - 1] for i, j in enumerate(length_s)]).view(len(length_s), self.hidden_dim)

        # SEQUENCE net
        self.hidden = self.init_hidden(batch_s)
        # embed your sequences
        embedding_seq = self.word_embeddings(inputs)
        # pack them up nicely
        packed_input = pack_padded_sequence(embedding_seq, sequences_len)
        # now run through LSTM
        packed_output, (self.hidden, ct) = self.lstm(packed_input, self.hidden)
        # undo the packing operation
        output_seq, length_s = pad_packed_sequence(packed_output)
        output_seq = output_seq.transpose(0, 1)
        seq_to_concat = torch.cat([output_seq[i, j.data - 1] for i, j in enumerate(length_s)]).view(len(length_s), self.hidden_dim)

        # MLP
        combined_tensor = torch.cat((seq_to_concat, pep_to_concat), 1)
        l1 = self.relu(self.hidden1(combined_tensor))
        score = self.hidden2tag(l1)
        soft_score = F.sigmoid(score)

        return soft_score

    def name_model(self):
        return 'Two embedding matrix '+str(self.word_embeddings)+' '+str(self.hidden_dim)+' '


class DoubleLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, vocab_size2, device_):
        super(DoubleLSTMClassifier, self).__init__()
        # GPU
        self.device = device_
        self.hidden_dim = hidden_dim
        # embedding matrix
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_embeddings_2 = nn.Embedding(vocab_size2, embedding_dim, padding_idx=0)
        # LStm
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)
        self.lstm_pep = nn.LSTM(embedding_dim, hidden_dim, num_layers=1)
        # The linear layer that connected to dense
        self.hidden1 = nn.Linear(hidden_dim + hidden_dim, hidden_dim // 2)
        self.relu = torch.nn.LeakyReLU()
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim // 2, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, size_batch=1):
        if self.device.type != 'cpu':
            return (autograd.Variable(torch.zeros(1, size_batch, self.hidden_dim)).to(self.device),
                    autograd.Variable(torch.zeros(1, size_batch, self.hidden_dim)).to(self.device))
        else:
            return (autograd.Variable(torch.zeros(1, size_batch, self.hidden_dim)),
                    autograd.Variable(torch.zeros(1, size_batch, self.hidden_dim)))

    def forward(self, inputs, sequences_len, input_pep, peptides_len):
        batch_s = input_pep.size()[-1]
        # PEPTIDE net
        self.hidden = self.init_hidden(batch_s)
        # embed your sequences
        embedding_pep = self.word_embeddings_2(input_pep)
        # pack them up nicely
        packed_input = pack_padded_sequence(embedding_pep, peptides_len)
        # now run through LSTM
        packed_output, (self.hidden, ct) = self.lstm_pep(packed_input, self.hidden)
        # undo the packing operation
        output_pep, length_s = pad_packed_sequence(packed_output)
        output_pep = output_pep.transpose(0, 1)
        # taking the last LSTM cell for each
        pep_to_concat = torch.cat([output_pep[i, j.data - 1] for i, j in enumerate(length_s)]).view(len(length_s), self.hidden_dim)

        # SEQUENCE net
        self.hidden = self.init_hidden(batch_s)
        # embed your sequences
        embedding_seq = self.word_embeddings(inputs)
        # pack them up nicely
        packed_input = pack_padded_sequence(embedding_seq, sequences_len)
        # now run through LSTM
        packed_output, (self.hidden, ct) = self.lstm(packed_input, self.hidden)
        # undo the packing operation
        output_seq, length_s = pad_packed_sequence(packed_output)
        output_seq = output_seq.transpose(0, 1)
        seq_to_concat = torch.cat([output_seq[i, j.data - 1] for i, j in enumerate(length_s)]).view(len(length_s), self.hidden_dim)

        # MLP
        combined_tensor = torch.cat((seq_to_concat, pep_to_concat), 1)
        l1 = self.relu(self.hidden1(combined_tensor))
        score = self.hidden2tag(l1)
        soft_score = F.sigmoid(score)

        return soft_score

    def name_model(self):
        return 'Two embedding matrix and two lstm'+str(self.word_embeddings)+' '+str(self.hidden_dim)+' '


class DoubleLSTMClassifierUpgrade(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, vocab_size2, device_, n_layers=1,
                 bi_lstm=False, drop=0):
        super(DoubleLSTMClassifierUpgrade, self).__init__()
        # GPU
        self.device = device_
        self.hidden_dim = hidden_dim
        self.num_layers = n_layers
        self.drop = drop
        self.bi_lstm = bi_lstm
        self.num_direction = 2 if bi_lstm else 1
        # embedding matrix
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_embeddings_2 = nn.Embedding(vocab_size2, embedding_dim, padding_idx=0)
        # LStm
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_layers, bidirectional=self.bi_lstm, dropout=self.drop)
        self.lstm_pep = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_layers, bidirectional=self.bi_lstm, dropout=self.drop)
        # The linear layer that connected to dense
        self.hidden1 = nn.Linear((hidden_dim + hidden_dim)*self.num_direction, hidden_dim // 2)
        self.relu = torch.nn.LeakyReLU()
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim // 2, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, size_batch=1):
        hidden_dim1 = self.num_direction * self.num_layers
        if self.device.type != 'cpu':
            return (autograd.Variable(torch.zeros(hidden_dim1, size_batch, self.hidden_dim)).to(self.device),
                    autograd.Variable(torch.zeros(hidden_dim1, size_batch, self.hidden_dim)).to(self.device))
        else:
            return (autograd.Variable(torch.zeros(hidden_dim1, size_batch, self.hidden_dim)),
                    autograd.Variable(torch.zeros(hidden_dim1, size_batch, self.hidden_dim)))

    def forward(self, inputs, sequences_len, input_pep, peptides_len):
        batch_s = input_pep.size()[-1]
        # PEPTIDE net
        self.hidden = self.init_hidden(batch_s)
        # embed your sequences
        embedding_pep = self.word_embeddings_2(input_pep)
        # pack them up nicely
        packed_input = pack_padded_sequence(embedding_pep, peptides_len)
        # now run through LSTM
        packed_output, (self.hidden, ct) = self.lstm_pep(packed_input, self.hidden)
        # undo the packing operation
        output_pep, length_s = pad_packed_sequence(packed_output)
        output_pep = output_pep.transpose(0, 1)
        # taking the last LSTM cell for each
        pep_to_concat = torch.cat([output_pep[i, j.data - 1] for i, j in enumerate(length_s)])\
            .view(len(length_s), self.num_direction*self.hidden_dim)

        # SEQUENCE net
        self.hidden = self.init_hidden(batch_s)
        # embed your sequences
        embedding_seq = self.word_embeddings(inputs)
        # pack them up nicely
        packed_input = pack_padded_sequence(embedding_seq, sequences_len)
        # now run through LSTM
        packed_output, (self.hidden, ct) = self.lstm(packed_input, self.hidden)
        # undo the packing operation
        output_seq, length_s = pad_packed_sequence(packed_output)
        output_seq = output_seq.transpose(0, 1)
        seq_to_concat = torch.cat([output_seq[i, j.data - 1] for i, j in enumerate(length_s)])\
            .view(len(length_s), self.num_direction*self.hidden_dim)

        # MLP
        combined_tensor = torch.cat((seq_to_concat, pep_to_concat), 1)
        l1 = self.relu(self.hidden1(combined_tensor))
        score = self.hidden2tag(l1)
        soft_score = F.sigmoid(score)

        return soft_score

    def name_model(self):
        return 'Two embedding matrix and two lstm with {} num layers and {} directions'.format(self.num_layers,
                                                                                               self.num_direction)+\
               str(self.word_embeddings)+' '+str(self.hidden_dim)+' '
