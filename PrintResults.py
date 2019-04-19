import os
import pickle
import numpy

path = "results/"
files = []

for r, d, f in os.walk(path):
    for file in f:
        files.append(os.path.join(r, file))

results = []
for f in files:
    with open(f, 'rb') as file:
        network = pickle.load(file)
        results.append(network)

for r in results:
    print(r[0] + "Accuracy: " + r[1])
