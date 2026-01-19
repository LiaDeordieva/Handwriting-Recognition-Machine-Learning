import numpy as np

#probability how safe the number
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#prediction
def forward(X, w):
    weighted_sum = np.matmul(X, w)
    return sigmoid(weighted_sum)    #probability for each number

#classify numbers
def classify(X, w):
    y_hat = forward(X, w)   #prediction, 10 numbers for image
    labels = np.argmax(y_hat, axis=1)   #biggest number
    return labels.reshape(-1, 1)

#mistakes
def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)     #for the right class
    second_term = (1 - Y) * np.log(1 - y_hat)       #for the false class
    return -np.sum(first_term + second_term) / X.shape[0]       #loss

#gradient descent
def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]     #directon

#status
def report(iteration, X_train, Y_train, X_test, Y_test, w):
    matches = np.count_nonzero(classify(X_test, w) == Y_test)   #number of right results
    n_test_examples = Y_test.shape[0]      #total test images
    matches = matches * 100.0 / n_test_examples     #hit rate
    training_loss = loss(X_train, Y_train, w)     #mistakes
    print("%d - Loss: %.20f, %.2f%%" % (iteration, training_loss, matches))

#training
def train(X_train, Y_train, X_test, Y_test, iterations, lr):
    w = np.zeros((X_train.shape[1], Y_train.shape[1]))      #start
    for i in range(iterations):
        report(i, X_train, Y_train, X_test, Y_test, w)      #current status
        w -= gradient(X_train, Y_train, w) * lr     #finding weights
    report(iterations, X_train, Y_train, X_test, Y_test, w)     #final status
    return w

#loading MNIST
import mnist as data
w = train(data.X_train, data.Y_train,
          data.X_test, data.Y_test,
          iterations=200, lr=1e-5)
