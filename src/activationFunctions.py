
"""
Compute the sigmoid of z

Arguments:
z -- A scalar or numpy array of any size.

Return:
s -- sigmoid(z)
"""

import numpy as np 
 
def sigmoid(z):

    s = 1/(1+np.exp(-z))
    
    return s
