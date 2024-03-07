import numpy as np
from utils import onehot

class Layer:

    """
    Base class for layers in the neural network with forward and backward pass.
    """
    def __init__(self):
        
        return

    def forward(self,inputs):
        raise NotImplementedError

    def backward(self,grad):
        raise NotImplementedError
    
    def step_gd(self,alpha):
        """
        Performs a gradient descent step given learning rate.
        Assumes that the layer has a parameter dictionary "params" on the form

        params = {
            'w1': {         
                'w': w,         The parameter matrix
                'd': d,         The gradient of loss wrt the parameter matrix
                },
            'w2': {....},
            
        }
        where each parameter has a key 'w' for weights and 'd' for gradients.
        """
        for param in self.params:
            self.params[param]['w'] -= alpha*self.params[param]['d']




class Attention(Layer):
    def __init__(self, d, k, init_scale = 0.1):
        """
        d: depth
        k: input length (number of digits in x)
        """
        self.softmax = Softmax()

        #Initalize w-s
        w_Q = np.random.randn(k, d)*init_scale
        w_K = np.random.randn(k, d)*init_scale
        w_O = np.random.randn(k, d)*init_scale
        w_V = np.random.randn(k, d)*init_scale

        self.params = {'w_Q':{'w':w_Q,'d':d}, 
                       'w_K':{'w':w_K,'d':d}, 
                       'w_O':{'w':w_O,'d':d}, 
                       'w_V':{'w':w_V,'d':d}}
        return

        

    def forward(self,x):
        """
        Input: x, shape (b,d,n) where b, d always constant, but n <= n_max

        Output: x_l, shape (b,d,n), same as input
        """
        n = x.shape[-1]
        #Initialize D
        D = np.zeros( (n, n) )
        i1,i2 = np.tril_indices(n,-1)
        D[i1,i2] -= np.inf
        self.x = x
        # Adjusting param size to input
        n = x.shape[-1]

        for param in self.params:
            print(param)

        # get w-s from dictionary
        w_Q = self.params['w_Q']['w']
        w_K = self.params['w_K']['w']
        w_O = self.params['w_O']['w']
        w_V = self.params['w_V']['w']

        self.x_T = np.transpose(x, axes=(0,2,1))

        #prod = x.T @ self.w_Q.T @ self.w_K @ x
        prod = np.einsum('bad,ds,sq,bqk -> bak',self.x_T,w_Q.T, w_K, x, optimize=True)
        A = self.softmax.forward(prod + D)
        self.A = A
        print(f"Shape of A: {A.shape}")
        # prod2 = self.w_O.T @ self.w_V @ x @ self.A
        prod2 = np.einsum('dk, kd, bds, bsn -> bdn', w_O.T, w_V, x, A, optimize=True)
        x_l = x + prod2
        return x_l


    def backward(self,grad):
        #TODO: Gjør skikkelig
        """
        Your code here
        """
        w_Q = self.params['w_Q']['w']
        w_K = self.params['w_K']['w']
        w_O = self.params['w_O']['w']
        w_V = self.params['w_V']['w']

        # g_OV = self.w_V.T @ self.w_O @ grad remove later
        g_OV = np.einsum("dk,kd,bdn -> bdn", w_V.T, w_O, grad)
        print(f"g_OV: {g_OV.shape}")
        g_S_int = np.einsum("bnd, bdk -> bnk", self.x_T, g_OV)
        g_S = self.softmax.backward(g_S_int)
        print(f"g_S: {g_S.shape}")

        A_T = np.transpose(self.A,(0,2,1))
        g_S_T = np.transpose(g_S,(0,2,1))
        prod = np.einsum("bdn, bnm -> bdm", g_OV, A_T) # g_OV @  self.A.T
        prod2 = np.einsum("dk,ks,bsn,bnm -> bdm", w_K.T, w_Q, self.x, g_S) #self.w_K.T @ self.w_Q @ self.x @ g_S
        prod3 = np.einsum("dk, ks, bsn, bnm -> bdm", w_Q.T, w_K, self.x, g_S_T) # self.w_Q.T @ self.w_K @ self.x @ g_S.T
        bA_l = grad + prod + prod2 + prod3
        return bA_l
    


class Softmax(Layer):

    def __init__(self):
        #TODO: Finne ut om vi må sette noen parametre her.
        self.epsilon = 1e-8
        return

    
    def forward(self,x):
        #print(f"softmax input shape: {x.shape}")
        P = np.exp(x - x.max(axis=1, keepdims=True))
        self.P = P
        Q = np.sum(P, axis=1, keepdims=True)
        self.Q = Q
        z_l =  P / (Q + self.epsilon)
        self.z_l = z_l
        return z_l


    def backward(self,grad):
        print(f"z_l:{self.z_l.shape}")
        print(f"grad: {grad.shape}") # 6,8,6
        print(f"P: {self.P.shape}")
        print(f"Q: {self.Q.shape}")


        return grad * self.z_l - np.sum(grad * (self.P / (self.Q * self.Q + self.epsilon)), axis=0, keepdims=True) * self.P



class CrossEntropy(Layer):

    def __init__(self):
        """
        Your code here
        """
        self.epsilon = 1e-8
        return

        

    def forward(self, x, y):
        """
        Your code here
        """
        b, m, n = np.shape(x)
        #print(y.shape[-1])
        self.x_tilde = x
        x_trunc = self.x_tilde[:,:,-y.shape[-1]:] #Truncate x to be same size as y
        Y = onehot(y,m)
        print(f"onehotshape: {Y.shape}")
        print(f"x shape: {x.shape}")
        self.x = x_trunc
        self.Y = Y
        ones = np.ones(m)
        p = np.transpose(ones) @ np.multiply(x_trunc, Y)
        q = -np.log(p)
        return np.mean(q)


    def backward(self):
        n = self.x_tilde.shape[-1]
        Z = np.zeros_like(self.x_tilde)
        Z[:,:,-self.Y.shape[-1]:] = self.Y
        print(Z[:,:,0])
        return -1/n*(Z/(self.x_tilde+self.epsilon))
    


class LinearLayer(Layer):

    """
    Linear Layer
    """
    def __init__(self,input_size, output_size,init_scale = 0.1):
        """
        Constructor takes input size and output size of layer 
        and scale for the weights
        """

        #Initialize weights using a sample from the normal distribution
        #scaled with the init_scale
        self.w = np.random.randn(output_size,input_size)*init_scale
        self.params = {"w":{'w':self.w,
                            'd':np.zeros_like(self.w)}}
        

    def forward(self,x):
        """
        Computes the affine transformation of the forward pass
        Stores input for backwards pass and returns output y = Wx.

        x: input, array of shape (batch_size, input_size, n) = (b,d,n)
        y: output, array of shape (batch_size, output_size, n) = (b,o,n)
        """

        # NOTE: Stores for backward
        self.x = x
        
        #Return output of layer
        #y = w@x
        y = np.einsum('od,bdn->bon',self.params["w"]['w'],x)
        return y
        
    def backward(self,grad):
        """
        Performs backward pass.

        grad: gradient of loss wrt output of layer, shape (batch_size, output_size, n) = (b,o,n)
        """

        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt weight w: 
        #dL/dw = (1/B)*sum_b^B (grad_b@x_b^T)
        self.params['w']['d'] = np.einsum('bon,bdn->od',grad,self.x)/b

        #Return gradient of loss wrt input of layer
        #dL/dw = w@grad.T
        return np.einsum('od,bon->bdn',self.params['w']['w'],grad)
    

class Relu(Layer):
    """
    Relu activation function
    """

    def __init__(self):
        return

    def relu(self,x):
        #relu(x) = max(0,x)
        return np.maximum(np.zeros(x.shape), x)

    def forward(self,x):
        
        #Store input for backwards pass
        self.x = x
        return self.relu(x)

    def backward(self,grad):

        #dL/dx = grad * relu'(x)
        return grad * np.where(self.x > 0, np.ones_like(self.x), np.zeros_like(self.x))



class EmbedPosition(Layer):
    def __init__(self,n_max,m,d,init_scale=1e-1):   

        """
        n_max: maximum length of input sequence
        m: number of items in the vocabulary / number of integers
        d: embedding dimension
        """

        #Initialize a linear layer for the embedding
        self.embed = LinearLayer(m,d,init_scale)
        #Initialize the position embedding matrix
        self.w = np.random.randn(d,n_max)*init_scale

        #Initialize the parameter dictionary for weight with key "Wp"
        self.params = {"Wp":{'w':self.w,'d':None}}

    def forward(self,X):

        """
        Input:
            X: one-hot encoded array of shape (b,m,n).

        Output:
            z_0: array of shape (b,d,n)

        embed.forward(X) maps (b,m,n) to (b,d,n). 
        Assigns a column of size d to each integer in the sequence
        and add positional embedding matrix (params['Wp']['w'][:,:n]) (b,d,n).

        Equivalent to 

        z_0 = W_E@X + W_P[:,:n]

        """

        #We assume that n < n_max
        n = X.shape[-1]
        z_0 = self.embed.forward(X) + self.params['Wp']['w'][:,:n]
        return z_0
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - None
        """

        
        b = grad.shape[0]

        


        #Compute gradient (average over B batches) of loss wrt positional embedding w:
        self.params['Wp']['d'] = np.zeros_like(grad[0]) # Kanskje dirty fix
        print(f"wpd shape: {self.params['Wp']['d'].shape}")
        print(f"grad shape: {grad.shape}")
        self.params['Wp']['d'] += np.sum(grad,axis=0)/b

        #Use backwards pass of the linear layer
        self.embed.backward(grad)

        #This is always the final layer, so we return None
        return None
    
    def step_gd(self,step_size):

        #We need to call the step_gd method of the linear layer
        self.embed.step_gd(step_size)

        #And since we override step_gd(), we use super 
        #which calls the step_gd() of the base class
        #and does gd for the paramters in the params dict
        super().step_gd(step_size)




class FeedForward(Layer):


    def __init__(self,d, p,init_scale = 0.1):
        """
        Input:
            d: input dimension of first layer and output of second
            p: output dimension of first and input of second.

        """

        #first linear layer with input size d and output size p
        self.l1 = LinearLayer(d,p,init_scale)

        #We use the Relu activation function
        self.activation = Relu()

        #second linear layer with input size p and output size d
        self.l2 = LinearLayer(p,d,init_scale)


    def forward(self,x):
        """
        Input:
            - x of shape (b,d,n)
        Output:
            - shape (b,d,n)

        This is equivalent to
        y = x + W2.T@Relu(W1@x)

         (W1,W2 are p x d)
        """

        self.x = x

        return x + self.l2.forward(self.activation.forward(self.l1.forward(x)))
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - derivative of loss wrt input x. Shape (b,d,n)
        
        """

        #We use backward pass of the linear layers and activation.
        #Recall that the backward pass reverse the order of the layers. 
        grad_feed_forward = self.l1.backward(self.activation.backward(self.l2.backward(grad)))

        #Since forward pass is x + W2.T@Relu(W1@x)
        return grad + grad_feed_forward


    def step_gd(self,step_size):

        #Call the step_gd method of the linear layers
        self.l1.step_gd(step_size)
        self.l2.step_gd(step_size)