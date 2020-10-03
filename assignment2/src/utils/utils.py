#!/usr/bin/env python

import numpy as np

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    ### YOUR CODE HERE
    # https://stackoverflow.com/a/52781450/8623391
    ### END YOUR CODE
    return x

# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. 

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    ### YOUR CODE HERE
    x = np.exp(x - np.max(x))
    x = x / x.sum(axis=0)
    ### END YOUR CODE
    return x
