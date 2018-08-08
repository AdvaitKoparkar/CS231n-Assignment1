import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W) # this is the prediction of training sample i, for each class
    scores -= np.max(scores)
    # calculate the probabilities that the sample belongs to each class
    probabilities = np.exp(scores) / np.sum(np.exp(scores))
    # loss is the log of the probability of the correct class
    loss += -np.log(probabilities[y[i]])

    probabilities[y[i]] -= 1 # calculate p-1 and later we'll put the negative back

    # dW is adjusted by each row being the X[i] pixel values by the probability vector
    for j in xrange(num_classes):
      dW[:,j] += X[i,:] * probabilities[j]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  W = W.T
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[0]

  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # scores = np.dot(X, W)
  # exp_scores = np.exp(scores)
  # sum_exp_scores = np.sum(exp_scores, axis=1)
  # log_exp_scores = np.log(exp_scores)
  # true_log_exp_score = log_exp_scores[np.arange(N), y.T]
  # log_sum_exp_scores = np.log(sum_exp_scores)
  #
  # loss_mat = -1*true_log_exp_score + log_sum_exp_scores
  #
  # loss = np.sum(loss_mat, axis=0) / N
  # loss = loss + 0.5*reg*np.sum(W**2)


  scores = np.dot(W, X.T)
  dscores = X.T

  exp_scores = np.exp(scores)
  sum_exp_scores = np.sum(exp_scores, axis = 0)

  loss_mat = -1*scores[y.T, np.arange(N)] + np.log(sum_exp_scores)
  dloss = exp_scores/sum_exp_scores
  dloss[y, np.arange(N)] = dloss[y, np.arange(N)] - 1

  loss = np.mean(loss_mat) + 0.5*reg*np.sum(W**2)

  dW = np.dot(dloss,dscores.T) / N + reg*W
  dW = dW.T
  W = W.T


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
