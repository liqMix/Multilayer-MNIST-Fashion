# Brady Young
# Used to print the network parameters and results

import os
import numpy as np

path = "results/"
files = []

# Create a list of all files
for r, d, f in os.walk(path):
    for file in f:
        files.append(os.path.join(r, file))

results = []

# For each file, load it into the results list
for f in files:
    with open(f, 'rb') as file:
        network = np.load(file)
        results.append(network)

# For each result, print the title, accuracy, and percentage
for r in results:
    print(r[0])
    print("Accuracy: ", r[1][0], "/", r[1][1], "\t\tPercentage: ", r[1][0]/r[1][1], "\n")
