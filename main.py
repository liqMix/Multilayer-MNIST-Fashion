from Network import *
from HyperParams import *
from Dataset import *
import pickle
dataset = Dataset()
results = []

activations = ["sig", "arctan"]
minibatch_sizes = [1, 10, 50, 100]
learning_rates = [0.01, 0.1, 5]
num_hidden = [0, 10, 50, 100, 500, 1000]


def write_network_results(n):
    with open("results/ACT" + n.params.activation + \
              "-LR"+str(n.params.rate).replace('.', 'd') + \
              "-HID"+str(n.params.num_hidden) + \
              "-BAT"+str(n.params.batch_size) + \
              ".pickle", "wb") as file:
        pickle.dump([network.params, network.accuracy, network.confMat], file)


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


for n_h in num_hidden:
    params = Params()
    params.num_hidden = n_h
    params.rate = 0.1
    params.batch_size = 64
    network = Network(dataset, params)
    network.run()
    write_network_results(network)

