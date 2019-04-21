# Brady Young
# This file provides a container for network parameters, with defaults if not provided
# parameters.


class Params():
    def __init__(self, num_hidden=0, rate=1, momentum=1, num_epochs=30, batch_size=1, activation="sig"):
        self.num_hidden = num_hidden
        self.rate = rate
        self.momentum = momentum
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.activation = activation

    def __repr__(self):
        out = "Number of Hidden Nodes: " + str(self.num_hidden) + "\t\t"\
              "Learning Rate: " + str(self.rate) + "\t\t"\
              "Batch Size: " + str(self.batch_size) + "\t\t"\
              "Activation: " + self.activation

        return out
