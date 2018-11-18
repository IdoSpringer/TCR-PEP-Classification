import matplotlib.pyplot as plt

loss = []

with open('epoch_loss', 'r') as file:
    for i in range(1200):
        line = file.readline()
    for j in range(1):
        loss.append([])
        for i in range(300):
            line = file.readline()
            loss_value = float(line.split()[-1])
            loss[j].append(loss_value)
        plt.plot(range(300), loss[j], 'bo')
        plt.ylim(0.5, 0.8)

plt.title("loss (BCE) per epoch, 3 peptides")
plt.show()