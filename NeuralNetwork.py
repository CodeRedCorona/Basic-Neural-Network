"""
This program implements a two layer neural network which learns a pattern in the given matrix.
"""


#Dependencies 
import numpy as np

#This class implements the Neural Network.
class NeuralNetwork():
    
	def __init__(self):
		#The random numbers are seed so that same numbers are produced every time the program is run.
		np.random.seed(1)
		
		#This matrix contains the weights of the connections between layer0 and layer1.
		#Since, layer0 contains 3 elements (inputs ) and layer1 contains 4 elements, we have a 3x4 matrix.
		self.synaptic_weights_0 = 2 * np.random.random((3, 4)) - 1
		
		#This matrix contains the weights of the connections between layer1 and layer2.
		#Since, layer1 contains 4 elements and layer2 contains only 1 element, we have a 4x1 matrix.
		self.synaptic_weights_1 = 2 * np.random.random((4, 1)) - 1
  
    def __sigmoid(self, x):
		#This function normalizes the value to a number between 0 and 1.
		#This function converts the numbers to probabilities. 
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
		#This function returns the derivative of the function. 
		#Essentially, it gives us the slope of the sigmoid function at any given point 
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        #This is where the learning takes place.
	
		for iteration in xrange(number_of_training_iterations):
			
			#Initializing the layers of the Neural Network and linking it with synaptic weights. 
			layer1 = self.think(training_set_inputs, self.synaptic_weights_0)
			
			layer2 = self.think(layer1, self.synaptic_weights_1)
			
			#The error shows how far away is our prediction from the actual value.
			l2_error = training_set_outputs - layer2
			
			#This step displays the mean value of error at the specified intervals.
			if (iteration % 10000) == 0:
				print "Error:" + str(np.mean(np.abs(l2_error)))
			
			#The change to be made in the weights of the synapses b/w layer 1 and layer 2.
			l2_delta = l2_error * self.__sigmoid_derivative(layer2)
			
			#This helps us determine how much each node in layer1 contributed to the error in layer2. AKA BACKPROPOGATION.
			l1_error = l2_delta.dot(self.synaptic_weights_1.T) 
			
			#The change to be made in the weights of the synapses b/w layer0 and layer1
			l1_delta = l1_error * self.__sigmoid_derivative(layer1)

			#UPDATING the weights of the synapses.
			self.synaptic_weights_1 += layer1.T.dot(l2_delta)
			
			self.synaptic_weights_0 += training_set_inputs.T.dot(l1_delta)


    def think(self, inputs, weights):

        return self.__sigmoid(np.dot(inputs, weights))


if __name__ == "__main__":


    neural_network = NeuralNetwork()

    inputs = np.array([[0, 0, 1],
		       [0, 1, 1],
		       [1, 0, 1],
		       [1, 1, 1]])

    output = np.array([[0],
		       [1],
		       [1],
		       [0]])

    neural_network.train(inputs, output, 100000)
