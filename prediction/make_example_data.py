import random

w = 'weizmann_pairs.txt'
with open(w, 'r') as file:
    pairs = []
    for line in file:
        pairs.append(line.strip().split())
# pairs = set(pairs)
sample = random.sample(pairs, 30)
for pair in sample:
    print(pair[0] + '\t' + pair[1])
pass