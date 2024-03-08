from neural_network import *
from layers import *
from training import trainModel
import numpy as np
from data_generators import get_train_test_addition, get_train_test_sorting
from training import *
import pickle

r = 5
m = 2
batchSize = 250
batches = 10
d = 10
k = 5
p = 15
L = 2
n_max = 2*r-1
sigma = Relu

data = get_train_test_sorting(r,m,batchSize, batches)

embed = EmbedPosition(n_max,m,d)
att1 = Attention(d,k)
att2 = Attention(d,k)
ff1 = FeedForward(d,p)
ff2 = FeedForward(d,p)
un_embed = LinearLayer(d,m)
softmax = Softmax()
loss = CrossEntropy()

a = np.copy(embed.params['Wp']['w'])

print(a[2][2])



nn = NeuralNetwork([embed,att1,ff1,att2, ff2,un_embed,softmax])

losses = trainModel(nn,data,3,loss, m, 0.001)
losses.shape

b = nn.layers[0].params['Wp']['w']

print(a[2,2], b[2,2])

