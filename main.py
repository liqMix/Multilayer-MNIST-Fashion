# Brady Young
# This is the main file to carry out network execution

from Network import *
from HyperParams import *
from Dataset import *
import numpy as np

# Possible network parameter values
activations = ["sig", "arctan"]
minibatch_sizes = [1, 10, 50, 100]
learning_rates = [0.01, 0.1, 5]
num_hidden = [10, 50, 100, 500, 1000]


# Writes the results of the network to a file for later reference
def write_network_results(n):
    with open("results/ACT" + n.params.activation + \
              "-LR"+str(n.params.rate).replace('.', 'd') + \
              "-HID"+str(n.params.num_hidden) + \
              "-BAT"+str(n.params.batch_size) + \
              ".npsave", "wb") as file:
        np.save(file, [n.params, n.accuracy, n.confMat])


# Create a dataset object and empty results list
dataset = Dataset()
results = []


# Create multiple networks with varying parameters
for a in activations:
    params = Params()
    params.activation = a

    for mb_s in minibatch_sizes:
        params.batch_size = mb_s

        for l_r in learning_rates:
            params.rate = l_r
            network = Network(dataset, params)
            network.run()
            write_network_results(network)

    params.rate = 0.1
    params.batch_size = 64
    for n_h in num_hidden:
        params.num_hidden = n_h
        network = Network(dataset, params)
        network.run()
        write_network_results(network)

