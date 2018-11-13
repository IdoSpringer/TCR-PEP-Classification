import matplotlib.pyplot as plt

names = ['train', 'dev', 'test', 'best test']
values = []
values35 = []
colors = ['gray', 'blue', 'red', 'green']
width = 0.35

with open("tests8.csv", 'r') as file:
    for k in range(7):  # Waiting for more result, should be 9
        values.append([])
        for i in range(4):
            line = file.readline()
            score = (line.split(","))[-2]
            values[k].append(float(score))
        _ = file.readline()

print(values)
print(list(range(1, 8)))

fig, axs = plt.subplots(3, 3)

# loop when all results are in file
axs[0, 0].bar(names, values[0], color=colors, width=width)
axs[0, 0].set_ylim(0.6, 0.8)
axs[0, 1].bar(names, values[0], color=colors, width=width)
axs[0, 1].set_ylim(0.6, 0.8)
axs[0, 2].bar(names, values[0], color=colors, width=width)
axs[0, 2].set_ylim(0.6, 0.8)
axs[1, 0].bar(names, values[0], color=colors, width=width)
axs[1, 0].set_ylim(0.6, 0.8)
axs[1, 1].bar(names, values[0], color=colors, width=width)
axs[1, 1].set_ylim(0.6, 0.8)
axs[1, 2].bar(names, values[0], color=colors, width=width)
axs[1, 2].set_ylim(0.6, 0.8)
axs[2, 0].bar(names, values[0], color=colors, width=width)
axs[2, 0].set_ylim(0.6, 0.8)


# plt.ylabel("accuracy")
# plt.title("Simple model")

fig.tight_layout()
plt.show()



# TODO Wait for more results and make grid-bar-graphs (3*3, each graph has 4*3 bars)
