import numpy as np
import gzip     #MNIST open
import struct   #read bytes

#probabiliry that the number is 5
def sigmoid(z):
    return 1 / (1 + np.exp(-z))         #0 or 1

#how safe the model
def forward(X, w):
    weighted_sum = np.matmul(X, w)      #matrix multiplication
    return sigmoid(weighted_sum)        #probability

#classify yes or no
def classify(X, w):
    return np.round(forward(X, w))      #0 or 1

#mistakes
def loss(X, Y, w):
    y_hat = forward(X, w)       #prediction
    first_term = Y * np.log(y_hat)      #case YES (1)
    second_term = (1 - Y) * np.log(1 - y_hat)       #case NO (0)
    return -np.average(first_term + second_term)        #smaller mistake - better model

#gradient descent
def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]     #direction to the best result

#training
def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))       #start
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))      #current loss
        w -= gradient(X, Y, w) * lr     #change the weight, lr - learning rate
    return w

#testing
def test(X, Y, w):
    total_examples = X.shape[0]     #number of total images
    correct_results = np.sum(classify(X, w) == Y)       #number of right predictions
    success_percent = correct_results * 100 / total_examples        #hit rate
    print("\nSuccess: %d/%d (%.2f%%)" %
          (correct_results, total_examples, success_percent))

#loading images (inputs)
def load_images(filename):
    with gzip.open(filename, 'rb') as f:        #open the file
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))      #read header
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)        #pixel as a number
        return all_pixels.reshape(n_images, columns * rows)

#bias
def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)       #add 1 at the front

#loading labels (outputs)
def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)       #skip header
        all_labels = f.read()       #read labels
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)     #single column matrix

#only number 5
def encode_fives(Y):
    return (Y == 5).astype(int)     #5 to 1, other to 0


#train images + bias
X_train = prepend_bias(load_images("../data/mnist/train-images-idx3-ubyte.gz"))
#test images
X_test = prepend_bias(load_images("../data/mnist/t10k-images-idx3-ubyte.gz"))
#train labels
Y_train = encode_fives(load_labels("../data/mnist/train-labels-idx1-ubyte.gz"))
#test labels
Y_test = encode_fives(load_labels("../data/mnist/t10k-labels-idx1-ubyte.gz"))


w = train(X_train, Y_train, iterations=100, lr=0.00001)
test(X_test, Y_test, w)

