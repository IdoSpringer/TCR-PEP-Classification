import numpy as np
import matplotlib.pyplot as plt

# file = 'myroc.npz'

# roc = np.load(file)
# plt.plot(roc['fpr'], roc['tpr'])
# plt.title('ROC curve, AUC=' + str(format(roc['auc'].item(), '.3f')))
# plt.savefig('ccccc')


def plot_roc(title, files, labels, colors):
    for file, label, color in zip(files, labels, colors):
        roc = np.load(file)
        plt.plot(roc['fpr'], roc['tpr'], label=label + ', AUC=' + str(format(roc['auc'].item(), '.3f')),
                 color=color)
    plt.title(title)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.show()
    pass

'''
plot_roc('Autoencoder model ROC curve, cancer data',
         ['ae_roc_exc2.npz', 'ae_roc_exc_gp2.npz'],
         ['internal negatives', 'external negatives'],
         ['dodgerblue', 'salmon'])


plot_roc('LSTM model ROC curve, cancer data',
         ['lstm_roc_exc2.npz', 'lstm_roc_exc_gp2.npz'],
         ['internal negatives', 'external negatives'],
         ['dodgerblue', 'salmon'])
'''
