from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    dW = np.zeros(W.shape)  # initialize the gradient as zero D by C

    # compute the loss and the gradient
    num_classes = W.shape[1] # C
    num_train = X.shape[0] # N
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) # X[i] is 1 by D and W is D by C, so the scores is 1 by C
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]: 
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i] # 当j != y[i]时，代价函数对权重W的偏导
                dW[:, y[i]] -= X[i] # 当j == y[i]时，代价函数对权重W的偏导

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train # 除以样本的总数 N
    dW += 2 * reg * W # 正则化项中对权重W的偏导

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W) # N by C
    num_train = X.shape[0] # N
    correct_class_score = scores[np.arange(num_train), y].reshape(num_train, 1) # N by 1

    margins = np.maximum(0, scores - correct_class_score + 1) # N by C
    # 真实分类的那一列置0，原本是1，保证正确分类不被计算进loss
    margins[np.arange(num_train), y] = 0
    # margins所有项求和并除以样本数，再加上正则化项就是loss
    loss = np.sum(margins) / num_train + reg * np.sum(W ** 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 梯度的计算需要转化成两个矩阵相乘
    dS = np.zeros_like(scores)
    # 这里的idx是二维坐标
    idx = np.where(scores - correct_class_score + 1 > 0)
    dS[idx] = 1;
    dS[np.arange(num_train), y] = -1 * (np.sum(scores - correct_class_score + 1 > 0, axis=1) - 1)
    dW = X.T.dot(dS) / num_train + 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
