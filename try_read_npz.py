import numpy as np
import matplotlib.pyplot as plt

file = 'myroc.npz'

roc = np.load(file)
plt.plot(roc['fpr'], roc['tpr'])
plt.title('ROC curve, AUC=' + str(format(roc['auc'].item(), '.3f')))
plt.savefig('ccccc')