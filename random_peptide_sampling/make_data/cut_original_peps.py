import pickle

with open('train sequences.pickle', 'rb') as train_file,\
     open('test sequences.pickle', 'rb') as test_file:
    train_list = pickle.load(train_file)
    cut_train_list = {}
    test_list = pickle.load(test_file)
    for pep in train_list:
        if len(train_list[pep]) > 600:
            cut_train_list[pep] = train_list[pep][:600]
        else:
            cut_train_list[pep] = train_list[pep]
    pickle.dump(cut_train_list, open('cut_train.pickle', 'wb'))


with open('cut_train.pickle', 'rb') as cut_train:
    list = pickle.load(cut_train)
    for pep in list:
        print(len(list[pep]))
