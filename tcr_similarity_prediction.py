from pair_sampling.plots import plot_mul_auc

def read_data():
    pass


def train_autoencoder():
    pass


def find_most_similar(tcr):
    pass


def predict_peptide(tcr):
    pass


# train TCR autoencoder

# given test TCR
# find the closest TCR (that binds to peptide)
# take this peptide for prediction
'''
plot_mul_auc(['ae_w_train_auc', 'ae_w_test_auc', 'w_train_auc_d_ed10_od10_lr0.001_wd0', 'w_test_auc_d_ed10_od10_lr0.001_wd0'],
             ['train tcr autoencoder', 'test tcr autoencoder', 'train tcr lstm', 'test tcr lstm'],
             'Different TCR encoding models on Weizmann data')



plot_mul_auc(['ae_w_train_auc_ep1000', 'ae_w_test_auc_ep1000',
              'ae_w_train_auc_ep1000_d01', 'ae_w_test_auc_ep1000_do01',
              'ae_w_train_auc_ep1000_d005', 'ae_w_test_auc_ep1000_d005'],
             ['train, dropout=0.5', 'test, dropout=0.5',
              'train, dropout=0.1', 'test, dropout=0.1',
              'train, dropout=0.05', 'test, dropout=0.05'],
             'TCR Autoencoder based model on Weizmann data with different dropouts')


plot_mul_auc(['ae_w_train_auc_ep200', 'ae_w_test_auc_ep200',
              'ae_w_train_auc_wd-3', 'ae_w_test_auc_wd-3',
              'ae_w_train_auc_wd-5', 'ae_w_test_auc_wd-5'],
             ['train, wd=0', 'test, wd=0',
              'train, wd=1e-3', 'test, wd=1e-3',
              'train, wd=1e-5', 'test, wd=1e-5'],
             'TCR Autoencoder based model on Weizmann data, different regularization')


plot_mul_auc(['ae_auc/ae_c_train_auc_ep1000', 'ae_auc/ae_c_test_auc_ep1000'],
             ['train', 'test'],
             'TCR Autoencoder based model on cancer data')
'''

plot_mul_auc(['ae_w_train_auc_wd-5_tr_ae', 'ae_w_test_auc_wd-5_tr_ae',
              'ae_w_train_auc_wd-4_tr_ae', 'ae_w_test_auc_wd-4_tr_ae'],
             ['train wd=1e-5', 'test wd=1e-5',
              'train wd=1e-4', 'test wd=1e-4'],
             'TCR Autoencoder based model with autoencoder training, Weizmann data')
