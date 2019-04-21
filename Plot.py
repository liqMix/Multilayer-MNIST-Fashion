# This file creates and saves a confusion matrix for each network,
# with each cell increasing in brightness according to their
# portion of the total

from matplotlib import pyplot as plt
from pandas import *
import numpy as np
import os

path = "results/"
files = []

for r, d, f in os.walk(path):
    for file in f:
        files.append(os.path.join(r, file))

results = []
for f in files:
    with open(f, 'rb') as file:
        network = np.load(file)
        results.append(network)


# Confusion Matrix representing error rates
def plot(n):
    fig = plt.figure()
    cell_height = 0.1
    cell_width = 0.1
    label = list(range(10))
    params = n[0]
    accuracy = n[1]
    conf_mat = n[2]

    df = DataFrame(conf_mat, index=label, columns=label)
    values = np.around(df.values, 2)
    norm = plt.Normalize(values.min() - 1, values.max() + 1)
    colours = plt.cm.hot(norm(values))

    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

    table = plt.table(cellText=values, rowLabels=label, colLabels=label,
                      colWidths=[0.03] * values.shape[1], loc='center',
                      cellColours=colours)

    ax.set_title("Learning Rate: " + str(params.rate) + \
                 " Batch Size: " + str(params.batch_size) + \
                 " Hidden Nodes: " + str(params.num_hidden), y=1.1)

    tc = table.properties()['child_artists']
    for cell in tc:
        cell.set_height(cell_height)
        cell.set_width(cell_width)

    ax.set_xlabel("Output", labelpad=-290)
    ax.set_ylabel("Target", labelpad=10)

    plt.savefig(path + "ACT" + params.activation +
                       "-LR"+str(params.rate).replace('.', 'd') +
                       "-HID"+str(params.num_hidden) +
                       "-BAT"+str(params.batch_size) + '.png', bbox_inches='tight')

    plt.close()


# Plot each result
path = "tables/"
for r in results:
    plot(r)