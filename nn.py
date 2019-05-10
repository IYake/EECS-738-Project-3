import random
import numpy as np

class NN(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def stochastic_gradient_desc(self, training_data, epochs, batch_size, learning_rate, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, learning_rate)
            if test_data:
                print("Epoch %d: %d / %d" % (j, self.accuracy(test_data), n_test))
            elif not test_data: 
                print("Epoch %d complete" % j)
                
    def feed(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
                
    def accuracy(self, test_data):
        test_results = [(np.argmax(self.feed(x)), y) for (x, y) in test_data]
        num_correct = 0
        for result in test_results:
            if result[0] == result[1]:
                num_correct += 1
        return num_correct
        #return sum(int(x == y) for (x, y) in test_results)
                
    def update_batch(self, batch, learning_rate):
        d_b = [np.zeros(b.shape) for b in self.biases]
        d_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in batch:
            d_d_b, d_d_w = self.backprop(x, y)
            d_b = [nb+dnb for nb, dnb in zip(d_b, d_d_b)]
            d_w = [nw+dnw for nw, dnw in zip(d_w, d_d_w)]
            
        self.weights = [w-(learning_rate/len(batch))*nw for w, nw in zip(self.weights, d_w)]
        self.biases = [b-(learning_rate/len(batch))*nb for b, nb in zip(self.biases, d_b)]
        
    def backprop(self, x, y):
        d_b = [np.zeros(b.shape) for b in self.biases]
        d_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        # store the activations for every layer / forward pass
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * d_sigmoid(zs[-1])
        d_b[-1] = delta
        d_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = d_sigmoid(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            d_b[-l] = delta
            d_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (d_b, d_w)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def d_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))