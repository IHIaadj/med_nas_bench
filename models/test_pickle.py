import pickle 

pick_path = './test.pickle'

configs = []
# dump a list to the pickle
with open (pick_path, 'rb') as pick:
    configs.append(pickle.load(pick))

print(configs)