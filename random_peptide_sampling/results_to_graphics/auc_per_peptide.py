import matplotlib.pyplot as plt

'''
num_of_peptides = list(range(2, 17))
print(num_of_peptides)
w_train_auc = [0.94027,0.88467,0.88388,0.9252,0.90288,0.88527,0.8779,0.90867,0.80438,0.85364,0.89196,0.91759,0.89299,0.88017,0.85428]
print(w_train_auc)
w_test_auc = [0.80947,0.81099,0.82072,0.84451,0.87283,0.85779,0.87118,0.87905,0.84423,0.88703,0.87864,0.89785,0.88136,0.89214,0.87608]
print(w_test_auc)

fig, ax = plt.subplots()
ax.plot(num_of_peptides, w_train_auc, label='train')
ax.plot(num_of_peptides, w_test_auc, label='test')
ax.set_ylim(0.6, 1.0)
plt.xticks(num_of_peptides)
ax.legend()
plt.title("AUC per number of peptides, Weizmann data")
plt.show()
'''

'''
num_of_peptides = list(range(2, 17))
print(num_of_peptides)
s_train_auc = [0.88477,0.86662,0.83238,0.87317,0.85516,0.8521,0.8393,0.88564,0.85752,0.86944,0.89144,0.88263,0.88982,0.88686,0.88416]
print(s_train_auc)
s_test_auc = [0.78042,0.77792,0.7957,0.8284,0.84581,0.8484,0.85939,0.86069,0.86487,0.86068,0.87778,0.87535,0.88212,0.87252,0.8808]
print(s_test_auc)

fig, ax = plt.subplots()
ax.plot(num_of_peptides, s_train_auc, label='train')
ax.plot(num_of_peptides, s_test_auc, label='test')
ax.set_ylim(0.6, 1.0)
plt.xticks(num_of_peptides)
ax.legend()
plt.title("AUC per number of peptides, Shugay data")
plt.show()
'''


num_of_peptides = list(range(2, 17))
print(num_of_peptides)
u_train_auc = [0.87809,0.82944,0.82252,0.78098,0.82988,0.81577,0.84376,0.82549,0.83788,0.82949,0.85314,0.85274,0.8114,0.84117,0.81472]
print(u_train_auc)
u_test_auc = [0.79522,0.79828,0.79602,0.77583,0.81262,0.81802,0.83746,0.85538,0.85135,0.84471,0.85391,0.86358,0.84896,0.84872,0.83206]
print(u_test_auc)

fig, ax = plt.subplots()
ax.plot(num_of_peptides, u_train_auc, label='train')
ax.plot(num_of_peptides, u_test_auc, label='test')
ax.set_ylim(0.6, 1.0)
plt.xticks(num_of_peptides)
ax.legend()
plt.title("AUC per number of peptides, Union data")
plt.show()
