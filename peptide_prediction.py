import torch
from autoencoder_model import PaddingAutoencoder
from train_autoencoder import get_batches, read_pred

def get_tcr_encoding(tcr, model):
    pass


def nearest_neighbor():
    pass


def predict_peptide():
    pass


if __name__ == '__main__':
    tcrs = ['CASSFGGAYEQYV'] * 10
    max_len = 25
    model = PaddingAutoencoder(max_len, 21, encoding_dim=30)
    checkpoint = torch.load('pad_autoencoder_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    amino_to_ix = checkpoint['amino_to_ix']
    ix_to_amino = checkpoint['ix_to_amino']
    model.eval()
    batch_size = 10

    batches = get_batches(tcrs, amino_to_ix, batch_size, max_len)
    padded_input = batches[0]
    concat = padded_input.view(batch_size, model.input_len * model.input_dim)
    print(model.encoder(concat))
    pass