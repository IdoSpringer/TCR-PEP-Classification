import torch
import numpy as np
import sys
from new_models import SiameseLSTMClassifier
from sklearn.manifold import MDS
import matplotlib.pyplot as plt


def embedding_scaling(model_file, embedding_dim, device):
    all_acids = set([letter for letter in 'ARNDCEQGHILKMFPSTWYV'])
    hydrophobic = set([letter for letter in 'ILVCAGMFYWHKT'])
    polar = set([letter for letter in 'YWHKREQDNST'])
    small = set([letter for letter in 'VCAGDNSTP'])

    model = SiameseLSTMClassifier(embedding_dim, 10, device)
    model_data = torch.load(model_file)
    model.load_state_dict(model_data['model_state_dict'])
    amino_to_ix = model_data['amino_to_ix']
    ix_to_amino = {index: amino for amino, index in amino_to_ix.items()}
    emb_matrix = model.state_dict()['embedding.weight']
    dist_matrix = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            dist = np.linalg.norm(emb_matrix[i+1] - emb_matrix[j+1])
            dist_matrix[i, j] = dist
    mds = MDS(n_components=2)
    scaling = mds.fit_transform(dist_matrix)

    # plot
    x = scaling[:, 0]
    y = scaling[:, 1]
    acid_types = [all_acids, hydrophobic, polar, small]
    labels = ['all', 'hydrophobic', 'polar', 'small']
    for acid_type, label in zip(acid_types, labels):
        fig, ax = plt.subplots()
        good = [i for i in range(20) if ix_to_amino[i+1] in acid_type]
        bad = [i for i in range(20) if ix_to_amino[i+1] not in acid_type]
        ax.scatter([x[i] for i in good], [y[i] for i in good], c='blue', label=label)
        ax.scatter([x[i] for i in bad], [y[i] for i in bad], c='red', label='not ' + label)
        for i in range(20):
            ax.annotate(ix_to_amino[i+1], (x[i] + 0.1, y[i] + 0.1))
        if acid_type != all_acids:
            ax.legend()
        plt.title('Embedding Distance Matrix Multidimensional Scaling. embedding_dim=' + str(embedding_dim))
        plt.show()
    pass


def main(argv):
    embedding_scaling(argv[1], int(argv[2]), argv[3])
    pass


if __name__ == '__main__':
    main(sys.argv)
