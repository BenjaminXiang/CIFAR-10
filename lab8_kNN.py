#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def get_distance(img_a, img_b):
    return np.sum(np.abs(np.subtract(img_a, img_b)))

def kNN(images_train, cls_idx_train, images_test):
    num_train = len(images_train)
    num_test = len(images_test)
    
    knn_idx = np.zeros(num_test, dtype=int)
    for i in range(num_test):
        test_img = images_train[i]
        min_distance = 3072
        for j in range(num_train):
            train_img = images_train[j]
            tmp_distance = get_distance(test_img, train_img)
            
            if tmp_distance < min_distance:
                min_distance = tmp_distance
                knn_idx[i] = cls_idx_train[j]
    
    return knn_idx 
            