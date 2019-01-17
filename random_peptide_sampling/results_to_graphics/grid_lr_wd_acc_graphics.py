import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

wds = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lrs = [1e-2, 1e-3, 1e-4]

names = ['precision', 'recall', 'F1-score']

precision = []
recall = []
f1_score = []
with open('grid_lr_wd_acc', 'r') as file:
    for i in range(len(wds)):
        precision.append([])
        recall.append([])
        f1_score.append([])
        for j in range(len(lrs)):
            precision[i].append([])
            recall[i].append([])
            f1_score[i].append([])
            next(file)
            for k in range(4):
                line = file.readline().split(',')
                if k % 2 == 0:
                    precision[i][j].append(float(line[-4]))
                    recall[i][j].append(float(line[-3]))
                    f1_score[i][j].append(float(line[-2]))
    print(precision)
    print(recall)
    print(f1_score)


rows = ['wd='+str(wd) for wd in wds]
cols = ['learning rate='+str(lr) for lr in lrs]

fig = plt.figure()
#gs = gridspec.GridSpec(len(wds), len(lrs))

fig, axs = plt.subplots(len(wds), len(lrs))

# plt.setp(axs.flat, ylabel='accuracy')


pad = 5  # in points

for ax, col in zip(axs[0], cols):
    ax.set_title(col)

for ax, row in zip(axs[:,0], rows):
    # ax.set_ylabel(row)
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='small', ha='right', va='center', rotation='vertical')


for i in range(len(wds)):
    for j in range(len(lrs)):
        X = list(range(3))
        width = 0.35
        # ax2 = plt.subplot(gs[i, j])
        acc_train = [precision[i][j][0], recall[i][j][0], f1_score[i][j][0]]
        acc_test = [precision[i][j][1], recall[i][j][1], f1_score[i][j][1]]
        # r1 = ax1.bar(names, acc_train, color='SkyBlue', label='train', width=0.25)
        # r2 = ax1.bar(names, acc_test, color='IndianRed', label='test', width=0.25)
        axs[i, j].bar(X, acc_train, color='IndianRed', width=width)
        axs[i, j].bar([sum(x) for x in zip(X,[width]*3)], acc_test, color='SkyBlue', width=width)
        # axs[i, j].set_xticks(X, ['precision', 'recall', 'F1_score'])
        axs[i, j].set_xticklabels(['sss', 'precision', 'recall', 'F1_score'])
        axs[i, j].set_ylim(0, 1)

fig.suptitle("Accuracy of 2 peptides with different learning rates and regularization")

fig.tight_layout()
plt.show()


'''
names = ['train', 'dev', 'test', 'best test']
values = []

colors = ['gray', 'blue', 'red', 'green']
width = 0.35

embedding_dims = [10, 35, 100]
hidden_dims = [10, 35, 100]

# (high dims are in tests8.csv)
# (low dims are in low_dims_3peps_all.txt)
with open("tests8.csv", 'r') as file:
    for i in range(3):
        values.append([])
        for j in range(3):
            values[i].append([])
            for k in range(4):
                line = file.readline()
                score = (line.split(","))[-2]
                values[i][j].append(float(score))
            _ = file.readline()

rows = ['embedding dim='+str(dim) for dim in embedding_dims]
cols = ['hidden dim='+str(dim) for dim in hidden_dims]

fig, axs = plt.subplots(3, 3)

plt.setp(axs.flat, ylabel='accuracy')

pad = 5  # in points

for ax, col in zip(axs[0], cols):
    ax.set_title(col)

for ax, row in zip(axs[:,0], rows):
    #  ax.set_ylabel(row)
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation='vertical')

for i in range(3):
    for j in range(3):
        axs[i, j].bar(names, values[i][j], color=colors, width=width)
        axs[i, j].set_ylim(0.65, 0.8)

fig.suptitle("High dimension embedding accuracy with 3 peptides")

fig.tight_layout()
plt.show()
'''


