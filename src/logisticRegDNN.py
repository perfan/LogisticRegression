import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import predict as pr
from PIL import Image
from scipy import ndimage
from dataLoader import load_dataset
from models import L_layer_model

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes= load_dataset()

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

### Initial input information ###

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))


### Flattening dataset for ease of use ###
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
test_x_flatten  = test_x_orig.reshape(test_x_orig.shape[0],-1).T

print ("train_set_x_flatten shape: " + str(train_x_flatten.shape))
print ("train_set_y shape: " + str(train_y.shape))
print ("test_set_x_flatten shape: " + str(test_x_flatten.shape))
print ("test_set_y shape: " + str(test_y.shape))
print ("sanity check after reshaping: " + str(train_x_flatten[0:5,0]))

### Normalizing dataset ###
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # num_px * num_px * 3
n_y = 1
layers_dims = [n_x, 20, 7, 5, n_y]

### Learning ###
parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate = 0.0075, num_iterations = 2500, print_cost = True)
pred_train = pr.predict(train_x, train_y, parameters)
pred_test = pr.predict(test_x, test_y, parameters)


### Plotting Learning Curve ###

