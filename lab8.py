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

def ANN(train_data, train_labels, input_dim, hidden_dim, output_dim):
    
    num_train_data = len(train_data)
    
    # w1 shape is (1024, h_dim)
    w1 = np.random.randn(input_dim, hidden_dim)*1e-2
    # b1 shape is (h_dim, 1)                       
    b1 = np.zeros( hidden_dim )                                                
    # w2 shape is (h_dim, 10)
    w2 = np.random.randn(hidden_dim, output_dim)*1e-2                       
    # b2 shape is (10, 1)
    b2 = np.zeros( output_dim )                                                
    
    input_learningrate = 0.2
    hidden_learningrate = 0.2
    
    for i in range(num_train_data):
        # Forward
        # train_data[i].shape is (1024, 1)
        # hidden_out shape is (hidden_dim, 1)
        hidden_out = forward(train_data[i], w1, b1)                            
        #output shape is (10, 1)
        output = forward(hidden_out, w2, b2)
        
        # formula 1-1
        # error shape is (10, 1)
        error = 1/2*((train_labels[i] - output)**2)                            
        #formula 1-2
        # w2_gradient shape is (10, 1)
        w2_gradient = error*dsigmoid(output)                                     
        #formula 1-3
        # w1_gradient shape is (hidden_dim, 1)
        w1_gradient = np.dot(w2, w2_gradient)*dsigmoid(hidden_out) 
        #Update w2 
        for j in range(output_dim):
            w2[:,j] += input_learningrate*w2_gradient[j]*hidden_out
        for j in range(hidden_dim):
            w1[:,j] += hidden_learningrate*w1_gradient[j]*train_data[i]
            
        #Update threshold
        b2 += hidden_learningrate*w2_gradient
        b1 += input_learningrate*w1_gradient
            
    return W1, b1, W2, b2    
        

def train(images, one_hot_labels):
    num_images = len(images)
    img_size = 32 
    
    # Data reduction
    # Convert an RGB image into Grayscale image 
    gray_images = np.dot(images[...,:3], [0.299, 0.587, 0.114])*255
    feature_vectors = np.zeros(shape=[num_images, img_size*img_size], dtype=float)
    
    # Get feature_vectors
    for i in range(num_images):
        feature_vectors[i] = images[i].ravel()
#    raise NotImplementedError


def predict(images):
    # Return a Numpy ndarray with the length of len(images).
    # e.g. np.zeros((len(images),), dtype=int) means all predictions are 'airplane's
    raise NotImplementedError
