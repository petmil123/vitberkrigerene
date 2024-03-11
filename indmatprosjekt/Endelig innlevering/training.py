from neural_network import *
from layers import *

def trainModel(nn: NeuralNetwork, data: dict, iterations: int, loss: Layer, m: int, slice_number: int, step_size: float = 0.01,verbose =False ) -> np.ndarray:
    """Inputs a neural network and required parameters for training, and performs training of the network.
    Returns an array with the mean loss of each iteration. If verbose is true, the value of the loss function gets printed
    each iteration."""

    xs = data['x_train']
    ys = data['y_train']

    batches = xs.shape[0]
    losses = np.zeros((iterations, batches))
    for j in range(iterations):
        for i in range(batches):
            x = xs[i]
            y = ys[i]

            X = onehot(x,m)
            Z = nn.forward(X)

            losses[j,i] = loss.forward(Z,y[:,-slice_number:]) 
            dLdZ = loss.backward()
            nn.backward(dLdZ)
            nn.step_adam(step_size, j+1)
        if verbose:
            print("Iterasjon ", str(j), " L = ",np.mean(losses[j,:]), " gradient = ", np.linalg.norm(dLdZ))

    return np.mean(losses, axis=1)

def predict(nn: NeuralNetwork, xs: dict, r:int, m:int):
    """Inputs a neural network and input values for the model, as well as required testing parameters.
    Outputs a prediction for each dataset"""

    batches = xs.shape[0]
    y = np.zeros((xs.shape[0],xs.shape[1],r))
    for i in range(batches):
        x = xs[i]
        for j in range(r):
            X = onehot(x,m)
            z = nn.forward(X)
            Z = np.argmax(z, axis=1)
            toAppend =Z[:,-1:]
            x = np.append(x, toAppend, axis=1)
        y[i,:,:] = x[:,-r:]

    return y