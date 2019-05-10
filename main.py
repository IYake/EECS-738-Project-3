import data
import nn

training_data, validation_data, test_data = data.load_data()
ann = nn.NN([784, 30, 20, 40, 10])

ann.stochastic_gradient_desc(training_data, 20, 100, 3.0, test_data=test_data)