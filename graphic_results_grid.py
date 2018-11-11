import matplotlib.pyplot as plt

names = ['train', 'dev', 'test', 'best test']
values = []
colors = ['gray', 'blue', 'red', 'green']
width = 0.35

with open("tests8.csv", 'r') as file:
    for i in range(4):
        line = file.readline()
        score = (line.split(","))[-2]
        values.append(float(score))


plt.bar(names, values, color=colors, width=width)
plt.ylabel("accuracy")
plt.ylim(0.6, 0.8)
plt.title("Embedding dim = 10, hidden dim = 10, simple model")
plt.show()
