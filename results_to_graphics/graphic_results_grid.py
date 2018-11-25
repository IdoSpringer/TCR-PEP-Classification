import matplotlib.pyplot as plt

names = ['train', 'dev', 'test', 'best test']
values = []

colors = ['gray', 'blue', 'red', 'green']
width = 0.35

embedding_dims = [10, 35, 100]
hidden_dims = [10, 35, 100]

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

rows = ['embedding dim'+str(dim) for dim in embedding_dims]
cols = ['hidden dim'+str(dim) for dim in hidden_dims]

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

fig.suptitle("First model, 3 peptides, changing embedding and hidden dims")

fig.tight_layout()
plt.show()

