import matplotlib.pyplot as plt

loss = []

file_name = "loss_lr-3_wd0_ep200"
number_of_peptides = 2
epochs = 100 * 2 * number_of_peptides

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
        # plt.ylim(0.5, 0.7)

plt.title("loss (BCE) per epoch, 2 fixed peptides, lr=, wd=")
plt.show()
