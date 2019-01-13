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

    plt.plot(list(range(len(losses))), losses)
    plt.title('Train loss  (Binary-Cross-Entropy)')
    plt.xlabel('number of examples')
    plt.ylabel('loss')
    plt.show()

    plt.plot(list(range(len(avg_losses))), avg_losses)
    plt.title('Train average loss  (Binary-Cross-Entropy)')
    plt.xlabel('number of examples / 100')
    plt.ylabel('average loss')
    plt.show()


plot_loss('loss_file1')
plot_loss('loss_file2')
plot_loss('loss_file3')




