import numpy as np
from random import shuffle
from past.builtins import xrange
import matplotlib.pyplot as plt

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    num_nz_classes = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]
        num_nz_classes += 1
    dW[:, y[i]] += -1*num_nz_classes*X[i]
  # Right now the loss is a sum over all training examples, but we want itShabd nahi hai n
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  # Make W of dimension C x D
  delta = 1.0
  W = W.T
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # Computing Scores for Entire Training Data (scores is of dimension N x C)
  scores = np.dot(X, W.T)
  # Find Margin for each class for each training data point
  margin = scores - scores[np.arange(X.shape[0])[:, None], y[:, None]] + delta
  # Set correct Class margin to zero
  margin[np.arange(X.shape[0])[:, None], y[:, None]] = 0
  # Find Loss for Data Set as average of loss for all training data points
  loss = np.sum(np.maximum(0, margin))
  loss /= X.shape[0]

  # Add Regularization Loss
  loss += 0.5*reg*np.sum(W*W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  Indicator = np.zeros(margin.shape)
  Indicator[margin > 0] = 1
  incorrect_counts = np.sum(Indicator, axis=1)
  Indicator[np.arange(X.shape[0]), y] = -incorrect_counts
  dW = np.dot(Indicator.T, X)

  dW = dW / X.shape[0]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  # Return W and dW to dimensions D x C
  W = W.T
  dW = dW.T
  return loss, dW
