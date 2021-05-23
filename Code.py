# -*- coding: utf-8 -*-

"""
@author: pmenaq
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
#https://matplotlib.org/stable/tutorials/introductory/customizing.html
plt.style.use('tableau-colorblind10')
from IPython .display import clear_output

# =============================================================================
# Crea dataset
# =============================================================================
n = 500
p = 2
X, Y = make_circles(n_samples=n, factor=0.5, noise=0.04)
Y = Y[:,np.newaxis]

# plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1],c="skyblue")
# plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1],c="salmon")
# plt.axis("equal")
# plt.show()

# =============================================================================
# Crea clase de capa de la red
# =============================================================================
class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f
        self.b = np.random.rand(1,n_neur)*2 - 1
        self.W = np.random.rand(n_conn,n_neur)*2 - 1


# =============================================================================
# Crea Funciones de activacion
# =============================================================================

sigm = (lambda x: 1/(1+np.exp(-x)),
        lambda x: x*(1-x))
relu = (lambda x:np.maximum(0,x),
        lambda x:np.where(x<0,0,1))

# _x = np.linspace(-5,5,100)
# plt.plot(_x,sigm[0](_x))
# plt.plot(_x,sigm[1](_x))

# =============================================================================
# crear red
# =============================================================================

def create_nn(topology,act_f):
    nn = []
    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l],topology[l+1], act_f))
    return nn

# =============================================================================
# Codigo de entrenamiento
# =============================================================================

l2_cost = (lambda yp,yr: np.mean((yp-yr)**2),
           lambda yp,yr: (yp-yr))


def train(neural_net, X, Y, l2_cost, lr=0.5, train=True):
    #forward pass
    out = [(None, X)]
    for l, layer in enumerate(neural_net):
        z = np.dot(out[-1][1],neural_net[l].W) + neural_net[l].b
        a = neural_net[l].act_f[0](z)
        out.append((z,a))
    # print(l2_cost[0](out[-1][1],Y))
    _W = None
    if train:
        deltas = []
        for l in reversed(range(0,len(neural_net))):
            z = out[l+1][0]
            a = out[l+1][1]
            #Calcula deltas
            if l == len(neural_net) - 1:
                deltas.insert(0,l2_cost[1](a,Y)*neural_net[l].act_f[1](a))
            else:
                deltas.insert(0,np.dot(deltas[0], _W.T )*neural_net[l].act_f[1](a))
            _W = neural_net[l].W
            # Implementa descenso del gradiente
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0],axis=0,keepdims=True)*lr
            neural_net[l].W = neural_net[l].W - np.dot(out[l][1].T, deltas[0]) * lr

    return out[-1][1]




topology = [p, 4 ,5,6,4,3 ,2, 1]
Lr = 0.05
neural_net = create_nn(topology,sigm)

loss=[]

fig, ax = plt.subplots(1,2,figsize=(12,5))
pY = train(neural_net,X,Y,l2_cost,lr=Lr)
loss_i = l2_cost[0](pY,Y)

for i in range(10000):
    pY = train(neural_net,X,Y,l2_cost,lr=Lr)
    loss_j = l2_cost[0](pY,Y)

    delta = loss_i/loss_j

    if delta <= 1e-5:
        break
    else:
        if i%25 ==0:
            loss.append(loss_j)
            loss_i = loss_j

            res = 50
            _x0 = np.linspace(-1.5,1.5,res)
            _x1 = np.linspace(-1.5,1.5,res)
            _Y  = np.zeros((res,res))
            for i0, x0 in enumerate(_x0):
                for i1,x1 in enumerate(_x1):
                    _Y[i0,i1] = train(neural_net, np.array([[x0,x1]]), Y, l2_cost, train = False)[0][0]
            ax[0].pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
            ax[0].axis("equal")
            ax[0].scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1],c="skyblue")
            ax[0].scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1],c="salmon")
            ax[1].plot(range(len(loss)),loss,c="k")
            plt.pause(0.5)
            clear_output(wait=True)








