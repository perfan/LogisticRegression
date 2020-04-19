import initialize as init
import optimize as opt
import predict as pr 
import numpy as np
import forwardPropagate as forprop
import backwardPropagate as backprop
import matplotlib.pyplot as plt


def logisticRegression(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
     """
     Builds the logistic regression model by calling the function you've implemented previously

     Arguments:
     X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
     Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
     X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
     Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
     num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
     learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
     print_cost -- Set to true to print the cost every 100 iterations

     Returns:
     d -- dictionary containing information about the model.
     """

     # initialize parameters with zeros (≈ 1 line of code)
     w, b = init.initialize_with_zeros(X_train.shape[0])

     # Gradient descent (≈ 1 line of code)
     parameters, grads, costs = opt.optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = False)
    
     # Retrieve parameters w and b from dictionary "parameters"
     w = parameters["w"]
     b = parameters["b"]
    
     # Predict test/train set examples (≈ 2 lines of code)
     Y_prediction_test = pr.Log_predict(w, b, X_test)
     Y_prediction_train = pr.Log_predict(w, b, X_train)

     # Print train/test Errors
     print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
     print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

     d = {"costs": costs,
          "grads": grads,
          "Y_prediction_test": Y_prediction_test, 
          "Y_prediction_train" : Y_prediction_train, 
          "w" : w, 
          "b" : b,
          "learning_rate" : learning_rate,
          "num_iterations": num_iterations}

     ### Plotting Learning Curve ###
     costs = np.squeeze(d['costs'])
     plt.plot(costs)
     plt.ylabel('cost')
     plt.xlabel('iterations (per hundreds)')
     plt.title("Learning rate =" + str(d["learning_rate"]))
     plt.savefig("./results/Learning_Curve.png")
     plt.show()
    
     return d


def two_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    parameters = init.initialize_parameters(n_x, n_h, n_y)
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        A1, cache1 = forprop.linear_activation_forward(X, W1, b1, activation = "relu")
        A2, cache2 = forprop.linear_activation_forward(A1, W2, b2, activation = "sigmoid")
        
        # Compute cost
        cost = forprop.compute_cost(A2, Y)
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = backprop.linear_activation_backward(dA2, cache2, activation = "sigmoid")
        dA0, dW1, db1 = backprop.linear_activation_backward(dA1, cache1, activation = "relu")
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        parameters = backprop.update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    print(layers_dims)
    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    parameters = init.initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = forprop.L_model_forward(X, parameters)
        
        # Compute cost.
        cost = forprop.compute_cost(AL, Y)
    
        # Backward propagation.
        grads = backprop.L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = backprop.update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.savefig("./results/Learning_Curve.png")
    plt.show()
    
    return parameters

