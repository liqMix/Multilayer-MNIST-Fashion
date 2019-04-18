import time
from HyperParams import *
from Node import *

## Controls the network functions and data ##
class Network:
    def __init__(self, dataset, params=None):
        if params is None:
            self.params = Params()
        else:
            self.params = params

        self.dataset = dataset
        self.trainIDX = None
        self.testIDX = None
        self.assignIndex()

        if self.params.num_hidden is not 0:
            self.hidden = Hidden(self.params.num_hidden, 785)
            self.output = Output(10, self.params.num_hidden + 1)
        else:
            self.hidden = None
            self.output = Output(10, 785)

        self.accuracy = 0.0
        self.confMat = np.zeros((10, 10))

    # Shuffle indexes to randomize train and test sets
    def assignIndex(self):
        idxs = list(range(len(self.dataset.data)))
        random.shuffle(idxs)

        self.trainIDX = idxs[:50000]
        self.testIDX = idxs[50000:]


    # Calls the training and testing functions
    # for each epoch and appends the results
    # to a list

    def run(self):
        print("Starting network...")
        print(self.params)
        start = time.clock()

        for i in range(self.params.num_epochs):
            self.train()

        self.accuracy = "%.5f" % self.test()  # from training and testing

        finish = time.clock() - start
        print("\nFinished in", "%.2f" % finish, "seconds.")
        print("Accuracy: ", self.accuracy)

    # Iterates through the training set:
    #   -Feeds the data through the network
    #   -Updates the network weights
    # Returns the accuracy       
    def train(self):
        # for each feature set in the training set v
        for i in self.trainIDX:
            train = self.dataset.data[i] # grab the featureset
            target = int(self.dataset.labels[i])  # set the appropriate target class
            self.feed(train)
            self.backprop(target, train, i)


    # Iterates through the test set:
    #   -Feeds the data through the network
    #   -Updates the confusion matrix
    # Returns the accuracy
    def test(self):
        correct = 0
        iterations = 0

        for i in self.testIDX:
            target = int(self.dataset.labels[i])
            test = self.dataset.data[i]

            result = int(self.feed(test))

            if result == target:
                correct += 1

            self.confMat[target][result] += 1  # update the confusion matrix
            iterations += 1

        return correct / iterations

    # Feeds the inputs through the network
    def feed(self, inputs):

        # Only use output layer if there are no hidden nodes
        if self.hidden is None:
            weights = self.output.weights * inputs
            sums = np.sum(weights, axis=1)
            self.output.activation(sums, self.params.activation)
            return np.argmax(self.output.activated)

        # Hidden Layer #
        weights = self.hidden.weights * inputs  # multiply each weight by its associated input
        sums = np.sum(weights, axis=1)  # sum the weighted input values
        self.hidden.activation(sums, self.params.activation)  # sets the hidden activation array to activated values of summed weights

        # Output Layer #
        weights = self.output.weights * self.hidden.activated
        sums = np.sum(weights, axis=1)
        self.output.activation(sums, self.params.activation)

        return np.argmax(self.output.activated)  # returns the index of largest value weight,
                                                 # which is the predicted class of input

    # Updates the network's weights by calculating error and propogating
    # the error backwards through the network    
    def backprop(self, target, features, epoch_num):
        target_array = np.full(10, 0.1)  # initializes array of target
        target_array[target] = 0.9  # values to 0.1, sets goal to 0.9

        # Only use output layer if there are no hidden nodes
        if self.hidden is None:
            self.output.calcError(target_array, self.params.activation)
            if((epoch_num % self.params.batch_size) is 0) and (epoch_num is not 0):
                self.output.update(features, self.params)
            return

        self.output.calcError(target_array, self.params.activation)
        self.hidden.calcError(self.output, self.params.activation)

        if((epoch_num % self.params.batch_size) is 0) and (epoch_num is not 0):
            self.hidden.update(features, self.params)
            self.output.update(self.hidden, self.params)

