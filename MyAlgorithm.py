import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

import gzip
import pickle

# Se define la función de activación y la derivada de la función de activación
def sigmoid(z):
    sig = 1.0/(1.0+np.exp(-z))
    return sig

def sigmoid_prime(z):
    sig_prime = sigmoid(z)*(1-sigmoid(z))
    return sig_prime

class Network(object):
    def __init__(self,sizes):
        # Sizes es una lista con el número de neuronas de cada capa
        # El número de capas esta dado por el tamaño especificado de la red
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Se inicializan los bias de cada neurona con valores random
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        # Se inicializan los pesos de conexiones entre neuronas con valores random
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        # Propagación hacia adelante, se calcula la activación utilizando la función sigmoide
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, train_data, epochs, mini_batch_size, eta, test_data = None):
        # Algoritmo para el descenso estocástico del gradiente
        training_data = list(train_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            # Se "barajan" los datos de entrenamiento
            random.shuffle(training_data)
            # Se crean mini lotes aleatorios de tamaño "mini_batch_size" a lo largo de todos los datos de entrenamiento
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {}".format(j, self.evaluate(test_data)*100/n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        # lista para almacenar las activaciones capa tras capa
        activations = [x]
        # Lista para almacenar los valores de z capa tras capa
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1],y) * sigmoid_prime(zs[-1])
        print(delta)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2,self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)

    def cost_derivative(self, output_activations, y):
        return(output_activations-y)
