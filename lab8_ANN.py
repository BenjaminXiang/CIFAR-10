import numpy as np
import matplotlib.pyplot as plt 

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def dsigmoid(y):
    return y*(1-y)

def forward(x, w, b):
    y = np.dot(w.T, x)+b
    out = sigmoid(y)
    return out 

def ANN(train_data, train_labels, input_dim, hidden_dim, output_dim):

    print("hidden_dim is : ", hidden_dim)
    
    num_train_data = len(train_data)
    
    # w1 shape is (1024, h_dim)
    w1 = np.random.randn(input_dim, hidden_dim)*1e-2
    # b1 shape is (h_dim, 1)                       
    b1 = np.zeros( hidden_dim )                                                
    # w2 shape is (h_dim, 10)
    w2 = np.random.randn(hidden_dim, output_dim)*1e-2                       
    # b2 shape is (10, 1)
    b2 = np.zeros( output_dim )                                                
    
    input_learningrate = 0.4
    hidden_learningrate = 0.4
    
    for k in range(3):
        input_learningrate *= 0.5
        hidden_learningrate *= 0.5
        for i in range(num_train_data):
            # Forward
            # train_data[i].shape is (1024, 1)
            # hidden_out shape is (hidden_dim, 1)
            hidden_out = forward(train_data[i], w1, b1)                            
            #output shape is (10, 1)
            output = forward(hidden_out, w2, b2)
        
            # formula 1-1
            # error shape is (10, 1)
            error = train_labels[i]-output                             
            #formula 1-2
            # w2_gradient shape is (10, 1)
            w2_gradient = error*dsigmoid(output)                                     
            #formula 1-3
            # w1_gradient shape is (hidden_dim, 1)
            w1_gradient = np.dot(w2, w2_gradient)*dsigmoid(hidden_out) 
            #Update w2 
            for j in range(output_dim):
                w2[:,j] += input_learningrate*w2_gradient[j]*hidden_out
            #Update w1
            for j in range(hidden_dim):
                w1[:,j] += hidden_learningrate*w1_gradient[j]*train_data[i]
            
            #Update threshold
            b2 += hidden_learningrate*w2_gradient
            b1 += input_learningrate*w1_gradient
            
    return w1, b1, w2, b2    

def get_gray_feature(images):
    
    num_images = len(images)
    img_size = 32
    
    # Data reduction
    # Convert an RGB image into Grayscale image 
    gray_images = np.dot(images[...,:3], [0.299, 0.587, 0.114])
    feature_vectors = np.zeros(shape=[num_images, img_size*img_size], dtype=float)
    
    # Get feature_vectors
    for i in range(num_images):
        feature_vectors[i] = gray_images[i].ravel()
        
    return feature_vectors
        

def train(images, one_hot_labels, images_test, cls_idx_test):
    num_images = len(images)
    img_size = 32 
    num_test = len(images_test)
    
    feature_vectors = np.zeros(shape=[num_images, img_size*img_size], dtype=float)
    
    feature_vectors = get_gray_feature(images)
    
    input_dim = 1024
    output_dim = 10
    
    w1_best, b1_best, w2_best, b2_best = ANN(feature_vectors, one_hot_labels, input_dim, 64, output_dim)
    test_idx = np.zeros(num_test, dtype=int)
    test_idx = predict(images_test, w1_best, b1_best, w2_best, b2_best)

    accuracy_rate_best = np.array((test_idx == cls_idx_test)).mean() 
    hidden_dim_best = 64

    print("hidden_dim %d accuracy_rate is %f, best is %f" % (64, accuracy_rate_best, accuracy_rate_best))
    x_la = [64]
    y_la = [accuracy_rate_best]
    
    tmp_idx = np.zeros(num_test, dtype=int)
    for hidden_dim in range(64+32, 1025, 32):
        w1, b1, w2, b2 = ANN(feature_vectors, one_hot_labels, input_dim, hidden_dim, output_dim)
        tmp_idx = predict(images_test, w1, b1, w2, b2)
        tmp_accuracy_rate = np.array((tmp_idx == cls_idx_test)).mean()
        
        print("hidden_dim %d accuracy_rate is %f, best is %f" % (hidden_dim, tmp_accuracy_rate, accuracy_rate_best))    
        x_la.append(hidden_dim)
        y_la.append(tmp_accuracy_rate)
        
        if tmp_accuracy_rate > accuracy_rate_best:
            w1_best = w1
            b1_best = b1
            w2_best = w2
            b2_best = b2
            accuracy_rate_best = tmp_accuracy_rate
            hidden_dim_best = hidden_dim
    
    plt.figure()
    plt.plot(x_la, y_la)
    plt.xlabel('number of hidden layer')   
    plt.ylabel('accuracy_rate')     
    print("accuracy_rate_best is:", accuracy_rate_best)
    print("hidden_dim_best is:", hidden_dim_best)    
    
    return w1_best, b1_best, w2_best, b2_best

def predict(images, w1, b1, w2, b2):
    # Return a Numpy ndarray with the length of len(images).
    # e.g. np.zeros((len(images),), dtype=int) means all predictions are 'airplane's
    img_size = 32
    num_images = len(images)
    
    feature_vectors = np.zeros(shape=[num_images, img_size*img_size], dtype=float)
    feature_vectors = get_gray_feature(images)    
    
    ann_idx = np.zeros(num_images, dtype=int)
    for i in range(num_images):
        hidden_out = forward(feature_vectors[i], w1, b1)
        output = forward(hidden_out, w2, b2)
        ann_idx[i] = np.argmax(output)
    
    return ann_idx


            
        
    