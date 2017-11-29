import numpy as np
import math


def sigmoid(x):
    length = len(x)
    ans = np.zeros(length)
    for i in range(length):
        ans[i] = 1.0/(1.0+math.exp(-x[i]))
    return ans

def dsigmoid(y):
    return y*(1-y)

def forward(x, w, b):
    out = sigmoid(np.dot(w.T, x)+b)
    return out 

def ANN(num_train_data, train_data, train_labels, input_dim, hidden_dim, output_dim, weight_scale=1e-3):
    
    W1 =np.random.randn(input_dim, hidden_dim)*weight_scale
    b1 = np.zeros( hidden_dim )
    W2 = np.random.randn(hidden_dim, output_dim)*weight_scale
    b2 = np.zeros( output_dim )
    
    
    for i in range(num_train_data):
        # Forward
        hidden_out = forward(train_data[i], W1, b1)
        output = forward(hidden_out, W2, b2)
    
        loss = train_labels[i] - output
        
        g = dsigmoid(output)*loss
        dw2 = rate*g*output                                 #5-1
        db2 = -rate*g
        
        e = np.dot(W2.T, g)*dsigmoid(hidden_out)
        dw1 = rate*e*train_data[i]
        db1 = -rate*e
        
        W1 += dw1
        b1 += db1
        W2 += dw2
        b2 += db2
        
    
    return W1, b1, W2, b2    
        

def train(images, one_hot_labels):
    num_images = len(images)
    img_size = 32 
    
    # Data reduction
    # Convert an RGB image into Grayscale image 
    gray_images = np.dot(images[...,:3], [0.299, 0.587, 0.114])*255
    feature_vectors = np.zeros(shape=[num_images, img_size*img_size], dtype=float)
    for i in range(num_images):
        feature_vectors[i] = images[i].ravel()
#    raise NotImplementedError


def predict(images):
    # Return a Numpy ndarray with the length of len(images).
    # e.g. np.zeros((len(images),), dtype=int) means all predictions are 'airplane's
    raise NotImplementedError
