# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
import numpy as np
import time
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
from collections import defaultdict

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 重みの初期化
        #input_size:784, output_size:10, hidden_size:50
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)
        self.h =  defaultdict()
        #W1:750*50, b1:50, W2: 50*10, b2:10

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def updateparams(self):
        self.layers['Affine1'].W = network.params["W1"]
        self.layers['Affine1'].b = network.params["b1"]
        self.layers['Affine2'].W = network.params["W2"]
        self.layers['Affine2'].b = network.params["b2"]

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        #print("netward loss start")
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # forward
        #print("netward gradient start")
        self.loss(x, t)

        # backward
        #print("network backward start")
        dout = 1
        dout = self.lastLayer.backward(dout)

        #dout: 100*10
        
        layers = list(self.layers.values())
        #print("layers=", layers)

        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        #print("set gradient grads")
        return grads





weight_init_std=0.01
start = time.time()
x_train = weight_init_std * np.random.randn(600, 30)
fh = open("test.pkl", "rb")
y_train = pickle.load(fh)
fh.close()
#print(x_train.shape)
#print(y_train.shape)
train_loss_list = []
network = TwoLayerNet(30, 20, 10)

for i in range(10000):
    #start = time.time()
    batch_mask = np.random.choice(600, 200)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]
    grad =  network.gradient(x_batch, y_batch)

    for key in ("W1", "b1", "W2", "b2"):
        '''
        if key not in network.h:
            network.h[key]= 0
            network.h[key] += grad[key] * grad[key]
        else:
            network.h[key] += grad[key] * grad[key]
        network.params[key] -= 0.01 * grad[key] /(np.sqrt(network.h[key]) + 1e-7)
        '''
        network.params[key] -= 0.01 * grad[key]
    network.updateparams()

    loss = network.loss(x_train, y_train)
    #print(loss)
    train_loss_list.append(loss)
    #end = time.time()
    #elapsed = end - start
    #print("Time taken: ", elapsed)

print("train_loss_list=", train_loss_list)
end = time.time()
elapsed = end - start
print("Time taken: ", elapsed)