from layers import *

class NeuralNetwork():
    """
    Neural network class that takes a list of layers
    and performs forward and backward pass, as well
    as gradient descent step.
    """

    def __init__(self,layers):
        #layers is a list where each element is of the Layer class
        self.layers = layers
    
    def forward(self,x):
        #Recursively perform forward pass from initial input x
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self,grad):
        """
        Recursively perform backward pass 
        from grad : derivative of the loss wrt 
        the final output from the forward pass.
        """

        #reversed yields the layers in reversed order
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def step_gd(self,alpha):
        """
        Perform a gradient descent step for each layer,
        but only if it is of the class LinearLayer.
        """
        for layer in self.layers:
            #Check if layer is of class a class that has parameters
            if isinstance(layer,LinearLayer,EmbedPosition,FeedForward,Attention):
                layer.step_gd(alpha)
        return