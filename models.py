import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifierSimple(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, vocab_size2, device_):
        super(LSTMClassifierSimple, self).__init__()
        # GPU
        self.device = device_
        self.hidden_dim = hidden_dim
        # embedding matrix
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_embeddings_2 = nn.Embedding(vocab_size2, embedding_dim, padding_idx=0)
        # LStm
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True) ############ FIX BAG
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
