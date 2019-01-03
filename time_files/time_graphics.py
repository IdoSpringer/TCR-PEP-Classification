import matplotlib.pyplot as plt
import numpy as np

train_time = {}

file_name = "time_for_training_1"


with open(file_name, 'r') as file:
    for line in file:
        line = line.split()
        time = float(line[-1])
        # Get the number without ':' or ','
        npep = int(line[0].split(':')[1][:-1])
        # print(npep)
        try:
            train_time[npep].append(time)
        except KeyError:
            train_time[npep] = [time]

pep_num = []
pep_time = []
for npep in train_time:
    time_list = train_time[npep]
    mean = sum(time_list) / len(time_list)
    pep_num.append(npep)
    pep_time.append(mean / 60)



plt.bar(pep_num, pep_time)
plt.xlabel("Number of peptides")
plt.xticks(range(2, 17))
plt.ylabel('Average training time (minutes)')
# plt.ylim(0.5, 0.7)

# print(train_time)
print(pep_num, pep_time)
plt.title("Training time for peptides")
plt.show()
