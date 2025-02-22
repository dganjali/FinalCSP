from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import ssl
import certifi
import pickle

ssl._create_default_https_context = ssl._create_unverified_context
np.random.seed(19)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

m = x_train.shape[0]
m_test = x_test.shape[0]
layer_dims = [784, 400, 100, 10] 

W1 = np.random.randn(layer_dims[1], layer_dims[0]) * np.sqrt(2.0 / layer_dims[0])
W2 = np.random.randn(layer_dims[2], layer_dims[1]) * np.sqrt(2.0 / layer_dims[1])
W3 = np.random.randn(layer_dims[3], layer_dims[2]) * np.sqrt(2.0 / layer_dims[2])

b1 = np.zeros((layer_dims[1], 1))
b2 = np.zeros((layer_dims[2], 1))
b3 = np.zeros((layer_dims[3], 1))

parameters = {"W1":W1,
              "W2":W2,
              "W3":W3,
              "b1":b1,
              "b2":b2,
              "b3":b3}

with open("parameters.pkl", "rb") as f:
    parameters = pickle.load(f)

train_set = x_train
test_set = x_test
train_set = train_set.reshape(m, -1).T / 255.0
test_set = test_set.reshape(m_test, -1).T / 255.0

train_labels = np.zeros((10, m)) 
for i in range(m):
    train_labels[y_train[i], i] = 1

test_labels = np.zeros((10, m_test)) 
for i in range(m_test):
    test_labels[y_test[i], i] = 1

def relu(Z):
    return np.maximum(0, Z)

def back_relu(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis = 0, keepdims = True)

def fwd_prop(X, parameters):
    W1, W2, W3, b1, b2, b3 = parameters["W1"], parameters["W2"], parameters["W3"], parameters["b1"], parameters["b2"], parameters["b3"]
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)
    cache = {"Z1":Z1,
             "A1":A1,
             "Z2":Z2,
             "A2":A2,
             "Z3":Z3,
             "A3":A3}
    return A3, cache

def back_prop(X, Y, parameters, cache, lambd):
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]
    Z3 = cache["Z3"]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]
    
    m = X.shape[1]

    dA3 = A3 - Y
    dZ3 = dA3 
    dW3 = 1/m * np.dot(dZ3, A2.T) + lambd/m * W3
    db3 = 1/m * np.sum(dZ3, axis = 1, keepdims = True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * back_relu(Z2)
    dW2 = (1/m) * np.dot(dZ2, A1.T) + lambd/m * W2
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * back_relu(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T) + lambd/m * W1
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
        "dW3": dW3,
        "db3": db3
    }

    return grads

def grad_descent(parameters, grads, learning_rate):
    W1 = parameters["W1"] - learning_rate * grads["dW1"]
    b1 = parameters["b1"] - learning_rate * grads["db1"]
    W2 = parameters["W2"] - learning_rate * grads["dW2"]
    b2 = parameters["b2"] - learning_rate * grads["db2"]
    W3 = parameters["W3"] - learning_rate * grads["dW3"]
    b3 = parameters["b3"] - learning_rate * grads["db3"]
    new_parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3,
    }
    return new_parameters

def predict(train_set, parameters):
    A3, cache = fwd_prop(train_set, parameters)
    predictions = np.argmax(A3, axis=0)
    return predictions

def compute_cost(A3, train_labels, parameters, lambd):
    m = train_labels.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cost = -np.mean(np.sum(train_labels * np.log(A3 + 1e-8), axis=0)) + (lambd / (2 * m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    return cost

def train(train_set, train_labels, parameters, epochs = 501, learning_rate = 0.05, lambd = 2.0):

    for i in range(epochs):
        A3, cache = fwd_prop(train_set, parameters)
        gradients = back_prop(train_set, train_labels, parameters, cache, lambd)
        parameters = grad_descent(parameters, gradients, learning_rate)

        if i % 25 == 0:
            cost = compute_cost(A3, train_labels, parameters, lambd)
            predictions = np.argmax(A3, axis = 0)

            accuracy = np.mean(predictions == np.argmax(train_labels, axis = 0)) * 100
            print(f"Epoch {i}, Cost: {cost:.4f}, Accuracy: {accuracy:.2f}%")

    return parameters

def test(test_set, test_labels, parameters, lambd = 2.0):
    A3, cache = fwd_prop(test_set, parameters)
    cost = compute_cost(A3, test_labels, parameters, lambd)
    predictions = np.argmax(A3, axis = 0)

    accuracy = np.mean(predictions == np.argmax(test_labels, axis = 0)) * 100
    print(f"Cost: {cost:.4f}, Accuracy: {accuracy:.2f}%")
    return predictions

def predict(test_set, parameters):
    A3, cache = fwd_prop(test_set, parameters)
    prediction = np.argmax(A3, axis = 0)
    return prediction

if __name__ == "__main__":

    #parameters = train(train_set, train_labels, parameters)


    print("Train set:")
    test(train_set, train_labels, parameters)
    print("Test set:")
    predictions = test(test_set, test_labels, parameters)

    
    plt.rcParams.update({'font.size': 4})
    for i in range(100):  
        #plt.subplot(4, 4, 1 + i)
        plt.imshow(x_test[i], cmap=plt.get_cmap('gray'))
        plt.text(1, 2, "Label: " + str(predictions[i]), fontsize = 15, color = 'white')
        plt.show()
    

    with open("parameters.pkl", "wb") as f:
        pickle.dump(parameters, f)