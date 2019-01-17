import torch
import numpy as np
import sys
from new_models import SiameseLSTMClassifier
from sklearn.manifold import MDS
import matplotlib.pyplot as plt


def embedding_scaling(model_file, device):
    model = SiameseLSTMClassifier(10, 10, device)
    model_data = torch.load(model_file)
    model.load_state_dict(model_data['model_state_dict'])
    amino_to_ix = model_data['amino_to_ix']
    ix_to_amino = {index: amino for amino, index in amino_to_ix.items()}
    # print(ix_to_amino)
    # print(amino_to_ix)
    # Print model's state_dict
    # print("Model's state_dict:")
    for param_tensor in model.state_dict():
        # print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        pass
    emb_matrix = model.state_dict()['embedding.weight']
    # print(emb_matrix)
    dist_matrix = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            dist = np.linalg.norm(emb_matrix[i+1] - emb_matrix[j+1])
            dist_matrix[i, j] = dist
    # print(dist_matrix)
    # print(emb_matrix[amino_to_ix['F']])
    mds = MDS(n_components=2)
    # print(dist_matrix.shape)
    scaling = mds.fit_transform(dist_matrix)
    labels = [ix_to_amino[i+1] for i in range(20)]
    # print(labels)
    # print(scaling.shape)
    # print(scaling)

    fig, ax = plt.subplots()
    x = scaling[:, 0]
    y = scaling[:, 1]
    ax.scatter(x, y)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i]+0.1, y[i]+0.1))
    plt.title('Embedding Distance Matrix Multidimensional Scaling')
    plt.show()
    pass


def main(argv):
    embedding_scaling(argv[1], argv[2])
    pass


if __name__ == '__main__':
    main(sys.argv)
