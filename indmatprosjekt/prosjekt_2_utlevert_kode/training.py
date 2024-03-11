from neural_network import *
from layers import *


def trainModel(nn: NeuralNetwork, data: dict, iterations: int, loss: Layer, m: int, slice_number: int, step_size: float = 0.01, ) -> np.ndarray:
    """Inputs a neural network and required parameters for training, and performs training of the network.
    Returns an array with the mean loss of each iteration"""

    xs = data['x_train']
    ys = data['y_train']

    batches = xs.shape[0]
    for j in range(iterations):
        losses = np.zeros((iterations, batches))
        for i in range(batches):
            x = xs[i]
            y = ys[i]

            X = onehot(x,m)
            Z = nn.forward(X)

            #losses[j,i] = loss.forward(Z, y)
            losses[j,i] = loss.forward(Z,y[:,-slice_number:]) #lagt til [:, -slice_number:]
            dLdZ = loss.backward()
            nn.backward(dLdZ)
            nn.step_adam(step_size, j+1)
        print("Iterasjon ", str(j), " L = ",np.mean(losses[j,:]), " gradient = ", np.linalg.norm(dLdZ))
    return np.mean(losses, axis=1)

def predict(nn: NeuralNetwork, xs: dict, r:int, m:int):
    
    batches = xs.shape[0]
    y = np.zeros((xs.shape[0],xs.shape[1],r))
    for i in range(batches):
        x = xs[i]
        for j in range(r):
            print(j)
            X = onehot(x,m)
            z = nn.forward(X)
            Z = np.argmax(z, axis=1)
            print(Z.shape)
            print(x.shape)
            toAppend =Z[:,-1:]
            x = np.append(x, toAppend, axis=1)
        y[i,:,:] = x[:,:-r]
    return y


            
