import matplotlib.pyplot as plt


def plot_loss(loss_file):
    with open(loss_file, 'r') as file:
        losses = []
        avg_losses = []
        avg = 0
        index = 1
        for line in file:
            loss = float(line.strip())
            losses.append(loss)
            avg += loss
            index += 1
            if index % 100 == 0:
                avg_losses.append(avg / 100)
                avg = 0

    plt.plot(list(range(len(losses))), losses, 'bp')
    plt.title('Train loss  (Binary-Cross-Entropy)')
    plt.xlabel('number of examples')
    plt.ylabel('loss')
    plt.show()

    plt.plot(list(range(len(avg_losses))), avg_losses)
    plt.title('Train average loss  (Binary-Cross-Entropy)')
    plt.xlabel('number of examples / 100')
    plt.ylabel('average loss')
    plt.show()


def plot_auc(train_auc, test_auc):
    with open(train_auc, 'r') as file:
        tr_auc = []
        for line in file:
            tr_auc.append(float(line.strip()))
    with open(test_auc, 'r') as file:
        te_auc = []
        for line in file:
            te_auc.append(float(line.strip()))

    plt.plot(list(range(len(tr_auc))), tr_auc, label='train')
    plt.plot(list(range(len(te_auc))), te_auc, label='test')

    plt.title('AUC score per number of epochs')
    plt.xlabel('epoch. batch_size=100')
    plt.ylabel('auc score')
    plt.legend()
    plt.show()


def plot_mul_auc(auc_files, labels, title):
    for auc_file, label in zip(auc_files, labels):
        with open(auc_file, 'r') as file:
            aucs = []
            for line in file:
                aucs.append(float(line.strip()))
            plt.plot(list(range(len(aucs))), aucs, label=label)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('auc score')
    plt.legend()
    plt.show()


# plot_loss('loss_file1')
# plot_loss('loss_file2')
# plot_loss('loss_file3')
# plot_loss('loss_file100_n')

# plot_auc('train_auc', 'test_auc')
# plot_auc('train_auc_b50', 'test_auc_b50')
# plot_auc('train_auc_b100', 'test_auc_b100')
'''
plot_mul_auc(['test_auc_lr0.001_wd0', 'test_auc_lr0.001_wd0.001', 'test_auc_lr0.001_wd0.0001', 'test_auc_lr0.001_wd1e-05',
              'test_auc_lr0.0001_wd0', 'test_auc_lr0.0001_wd0.001', 'test_auc_lr0.0001_wd0.0001', 'test_auc_lr0.0001_wd1e-05',
              'test_auc_lr1e-05_wd0', 'test_auc_lr1e-05_wd0.001', 'test_auc_lr1e-05_wd0.0001', 'test_auc_lr1e-05_wd1e-05'],
             ['lr=1e-3, wd=0', 'lr=1e-3, wd=1e-3', 'lr=1e-3, wd=1e-4', 'lr=1e-3, wd=1e-5',
              'lr=1e-4, wd=0', 'lr=1e-4, wd=1e-3', 'lr=1e-4, wd=1e-4', 'lr=1e-4, wd=1e-5',
              'lr=1e-5, wd=0', 'lr=1e-5, wd=1e-3', 'lr=1e-5, wd=1e-4', 'lr=1e-5, wd=1e-5'])

plot_mul_auc(['train_auc_lr0.01_wd0', 'train_auc_lr0.01_wd0.001', 'train_auc_lr0.01_wd0.0001', 'train_auc_lr0.01_wd1e-05',
              'train_auc_lr0.1_wd0', 'train_auc_lr0.1_wd0.001', 'train_auc_lr0.1_wd0.0001', 'train_auc_lr0.1_wd1e-05'],
             ['lr=1e-2, wd=0', 'lr=1e-2, wd=1e-3', 'lr=1e-2, wd=1e-4', 'lr=1e-2, wd=1e-5',
              'lr=1e-1, wd=0', 'lr=1e-1, wd=1e-3', 'lr=1e-1, wd=1e-4', 'lr=1e-1, wd=1e-5'])
'''

plot_mul_auc(['train_auc_s', 'train_auc_d', 'test_auc_s', 'test_auc_d'],
             ['siamese model train', 'double model train', 'siamese model test', 'double model test'],
             'Train and test AUC for different models')
#fff