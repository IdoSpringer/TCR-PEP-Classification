import matplotlib.pyplot as plt

loss = []

file_name = "loss_1pep"
epochs = 100

with open(file_name, 'r') as file:
    for j in range(1):
        loss.append([])
        for i in range(epochs):
            line = file.readline()
            loss_value = float(line.split()[-1])
            loss[j].append(loss_value)
        plt.plot(range(epochs), loss[j], 'bo')
        plt.xlabel("epoch")
        plt.ylabel('loss')
        plt.ylim(0.0, 0.01)

plt.title("loss (BCE) per epoch, 1 peptide, lr=0.001")
plt.show()
