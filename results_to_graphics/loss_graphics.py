import matplotlib.pyplot as plt

loss = []

file_name = "loss_2pep"
epochs = 200

with open(file_name, 'r') as file:
    for j in range(1):
        loss.append([])
        for i in range(epochs):
            line = file.readline()
            loss_value = float(line.split()[-1])
            loss[j].append(loss_value)
        plt.plot(range(epochs), loss[j], 'bo')
        # plt.ylim(0.0, 0.9)

plt.title("loss (BCE) per epoch, 2 peptides, lr=0.001")
plt.show()
