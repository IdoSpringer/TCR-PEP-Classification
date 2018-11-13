import matplotlib.pyplot as plt

names = ['train', 'dev', 'test', 'best test']
values10 = []
values35 = []
colors = ['gray', 'blue', 'red', 'green']
width = 0.35

with open("tests8.csv", 'r') as file:
    for i in range(4):
        line = file.readline()
        score = (line.split(","))[-2]
        values10.append(float(score))
    for i in range(4):
        line = file.readline()
        score = (line.split(","))[-2]
        values35.append(float(score))


plt.figure(1)
plt.bar(names, values10, color=colors, width=width)
plt.ylabel("accuracy")
plt.ylim(0.6, 0.8)
plt.title("Embedding dim = 10, hidden dim = 10, simple model")


plt.figure(2)
plt.bar(names, values35, color=colors, width=width)
plt.ylabel("accuracy")
plt.ylim(0.6, 0.8)
plt.title("Embedding dim = 35, hidden dim = 10, simple model")

plt.show()

# TODO Wait for more results and make grid-bar-graphs (3*3, each graph has 4*3 bars)
