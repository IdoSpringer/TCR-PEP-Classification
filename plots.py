from pair_sampling.plots import plot_mul_auc


def max_auc(auc_file):
    with open(auc_file, 'r') as file:
        aucs = []
        for line in file:
            aucs.append(float(line.strip()))
        max_auc = max(aucs)
        last_auc = aucs[-1]
        print(auc_file)
        print('last auc:', last_auc)
        print('max auc:', max_auc)
    pass


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


plot_mul_auc(['ae_w_train_auc_wd-5_tr_ae', 'ae_w_test_auc_wd-5_tr_ae',
              'ae_w_train_auc_wd-4_tr_ae', 'ae_w_test_auc_wd-4_tr_ae'],
             ['train wd=1e-5', 'test wd=1e-5',
              'train wd=1e-4', 'test wd=1e-4'],
             'TCR Autoencoder based model with autoencoder training, Weizmann data')


plot_mul_auc(['ae_c_train_auc_wd-5_tr_ae', 'ae_c_test_auc_wd-5_tr_ae',
              'ae_c_train_auc_wd5-4_tr_ae', 'ae_c_test_auc_wd5-4_tr_ae',
              'ae_c_train_auc_wd-4_tr_ae', 'ae_c_test_auc_wd-4_tr_ae'],
             ['train, wd=1e-5', 'test, wd=1e-5',
              'train, wd=5e-4', 'test, wd=5e-4',
              'train, wd=1e-4', 'test, wd=1e-4'],
             'Autoencoder based model with trained parameters on cancer data, different regularizations')


plot_mul_auc(['ae_auc/ae_w_train_auc_wd-5_tr_ae', 'ae_auc/ae_w_test_auc_wd-5_tr_ae',
              'ae_w_train_auc_wd-5_gp', 'ae_w_test_auc_wd-5_gp'],
             ['train in domain negs', 'test in domain negs',
              'train out of domain negs', 'test out of domain negs'],
             'Autoencoder based model, Weizmann data, different negative samplings')


plot_mul_auc(['ae_s_train_auc_wd-5', 'ae_s_test_auc_wd-5',
              'ae_s_train_auc_wd-5_gp', 'ae_s_test_auc_wd-5_gp'],
             ['train in domain negs', 'test in domain negs',
              'train out of domain negs', 'test out of domain negs'],
             'Autoencoder based model, Shugay data, different negative samplings')


plot_mul_auc(['ae_c_train_auc_wd-5_tr_ae', 'ae_c_test_auc_wd-5_tr_ae',
              'ae_c2_train_auc_wd-5_gp', 'ae_c2_test_auc_wd-5_gp'],
             ['train in domain negs', 'test in domain negs',
              'train out of domain negs', 'test out of domain negs'],
             'Autoencoder based model, Cancer data, different negative samplings')


plot_mul_auc(['pair_sampling/train_w_d01_hdim30_wd-5', 'pair_sampling/test_w_d01_hdim30_wd-5',
              'lstm_w_train_auc_gp', 'lstm_w_test_auc_gp'],
             ['train in domain negs', 'test in domain negs',
              'train out of domain negs', 'test out of domain negs'],
             'LSTM based model, Weizmann data, different negative samplings')


plot_mul_auc(['pair_sampling/train_s_d01_hdim30_wd-5', 'pair_sampling/test_s_d01_hdim30_wd-5',
              'lstm_s_train_auc_gp', 'lstm_s_test_auc_gp'],
             ['train in domain negs', 'test in domain negs',
              'train out of domain negs', 'test out of domain negs'],
             'LSTM based model, Shugay data, different negative samplings')


plot_mul_auc(['pair_sampling/train_c_d01_hdim30_wd-5', 'pair_sampling/test_c_d01_hdim30_wd-5',
              'lstm_c_train_auc_gp', 'lstm_c_test_auc_gp'],
             ['train in domain negs', 'test in domain negs',
              'train out of domain negs', 'test out of domain negs'],
             'LSTM based model, Cancer data, different negative samplings')


plot_mul_auc(['nettcr_lstm_train', 'nettcr_lstm_test'],
             ['train', 'test'],
             'LSTM based model results on NetTCR data')



plot_mul_auc(['nettcr_ae_train', 'nettcr_ae_test'],
             ['train', 'test'],
             'Autoencoder based model results on NetTCR data')


plot_mul_auc(['nettcr_ae_train', 'nettcr_ae_test',
              'nettcr_ae_train_wd-4', 'nettcr_ae_test_wd-4'],
             ['train, wd=1e-5', 'test, wd=1e-5',
              'train, wd=1e-4', 'test, wd=1e-4'],
             'Autoencoder based model results on NetTCR data with different regularizations')


plot_mul_auc(['nettcr_lstm_train', 'nettcr_lstm_test',
              'nettcr_lstm_train_wd-4', 'nettcr_lstm_test_wd-4'],
             ['train, wd=1e-5', 'test, wd=1e-5',
              'train, wd=1e-4', 'test, wd=1e-4'],
             'LSTM based model results on NetTCR data with different regularizations')



plot_mul_auc(['ae_exc_train_wd-5', 'ae_exc_test_wd-5',
              'ae_exc_train_wd-4', 'ae_exc_test_wd-4'],
             ['train, wd=1e-5', 'test, wd=1e-5',
              'train, wd=1e-4', 'test, wd=1e-4'],
             'Autoencoder based model on extended cancer with different regularizations')


plot_mul_auc(['lstm_exc_train_wd-5', 'lstm_exc_test_wd-5',
              'lstm_exc_train_wd-4', 'lstm_exc_test_wd-4'],
             ['train, wd=1e-5', 'test, wd=1e-5',
              'train, wd=1e-4', 'test, wd=1e-4'],
             'LSTM based model on extended cancer with different regularizations')


plot_mul_auc(['ae_exsc_train_wd-5', 'ae_exsc_test_wd-5',
              'ae_exsc_train_wd-4', 'ae_exsc_test_wd-4'],
             ['train, wd=1e-5', 'test, wd=1e-5',
              'train, wd=1e-4', 'test, wd=1e-4'],
             'Autoencoder based model on (safe) extended cancer with different regularizations')


plot_mul_auc(['lstm_exsc_train_wd-5', 'lstm_exsc_test_wd-5',
              'lstm_exsc_train_wd-4', 'lstm_exsc_test_wd-4'],
             ['train, wd=1e-5', 'test, wd=1e-5',
              'train, wd=1e-4', 'test, wd=1e-4'],
             'LSTM based model on (safe) extended cancer with different regularizations')


plot_mul_auc(['lstm_exsc_train_wd-5_ep500', 'lstm_exsc_test_wd-5_ep500',
              'lstm_exsc_train_wd-4_ep500', 'lstm_exsc_test_wd-4_ep500'],
             ['train, wd=1e-5', 'test, wd=1e-5',
              'train, wd=1e-4', 'test, wd=1e-4'],
             'LSTM based model on (safe) extended cancer with different regularizations')


plot_mul_auc(['ae_exnos_train_wd-5', 'ae_exnos_test_wd-5',
              'lstm_exnos_train_wd-5', 'lstm_exnos_test_wd-5'],
             ['autoencoder train', 'autoencoder test',
              'lstm train', 'lstm test'],
             'Extended cancer data (without Shugay) with both models')
'''

# max_auc('ae_exnos_train_wd-5')
# max_auc('ae_exnos_test_wd-5')
# max_auc('lstm_exnos_train_wd-5')
# max_auc('lstm_exnos_test_wd-5')

# max_auc('nettcr_auc/nettcr_ae_train')
# max_auc('nettcr_auc/nettcr_ae_test')
# max_auc('nettcr_auc/nettcr_ae_train_wd-4')
# max_auc('nettcr_auc/nettcr_ae_test_wd-4')
# max_auc('nettcr_auc/nettcr_lstm_train')
# max_auc('nettcr_auc/nettcr_lstm_test')
# max_auc('nettcr_auc/nettcr_lstm_train_wd-4')
# max_auc('nettcr_auc/nettcr_lstm_test_wd-4')

# max_auc('ae_w_test_auc_gp_ep500')
# max_auc('ae_w_train_auc_gp_ep500')
# max_auc('lstm_w_test_auc_gp_ep500')
# max_auc('lstm_w_train_auc_gp_ep500')