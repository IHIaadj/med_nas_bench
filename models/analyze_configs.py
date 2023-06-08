import numpy as np 
import os 
import json 
import matplotlib.pyplot as plt
nb_params = []

for i in range(200):
    config = json.load(open("./configs/sample{}.json".format(i), 'r'))
    nb_params.append(config["params"])
    print(config['params'])

plt.hist(nb_params)
plt.show()

