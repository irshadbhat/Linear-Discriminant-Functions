#!/usr/bin/env python 
#!-*- coding: utf-8 -*-

"""
Binary Perceptron

Implementation of Single Sample Perceptron without Margin, Margin Infused 
Single Sample Perceptron (MISSP) and Margin Infused Relaxation Algorithm (MIRA).
"""

import numpy as np
import pylab as pl

__author__ = "Irshad Ahmad Bhat"
__version__ = "1.0"
__email__ = "irshad.bhat@research.iiit.ac.in"

class Perceptron():

    """
    classification algorithm based on a linear predictor function 
    combining a set of weights with the feature vector
    """

    def __init__(self, learning_rate = 0.1, margin = 0.0):

	self.margin = margin
	self.learning_rate = learning_rate
	
    def normalize_train(self, X, y):

	"""Convert input vectors to augmented vectors and normalize data 
	by changing sign of all data points in one of the two class."""	
	X = np.asarray(X)
	y = np.asarray(y)
	self.labels_ = np.unique(y)
    	# include bias unit 
    	self.train_set = self.augmented_set(X)
    	# choose one class as negative class
	self.idx = y==self.labels_[0]
	self.train_set[self.idx] = -self.train_set[self.idx]
	self.dim = self.train_set.shape[1] # trainset feature dimention
	
    def augmented_set(self, X):
	
    	# include bias unit 
	bias = np.ones((len(X), 1))
    	return np.hstack((X, bias))

    def MISSP_update(self):
	"""update weights whenever there is a mis-classification"""	
	for k in self.train_set:
	    if np.dot(self.weights, k) <= self.margin:
                self.weights += self.learning_rate * k
                self.flag = True

    def MIRA_update(self):
	"""update weights with smallest possible value to correct mistake"""	
	for k in self.train_set:
	    if np.dot(self.weights, k) <= self.margin:
    	        update_ = self.learning_rate*(self.margin - np.dot(self.weights, k))
    	        update_ *= k/(pl.norm(k)**2)
    	        self.weights += update_
                self.flag = True
		
    def MISSP(self, X, y):
	"""Margin Infused Single Sample Perceptron"""
    	self.flag = True
	self.normalize_train(X, y)
	self.update = self.MISSP_update
	self.fit()
    
    def MIRA(self, X, y):
	"""Margin Infused Relaxation Algorithm"""
    	self.flag = True
	self.normalize_train(X, y)
	self.update = self.MIRA_update
	self.fit()

    def fit(self):
	"""update weights until stopping criterion is reached"""	
    	itr = 0
	# initialize weights
	self.weights = np.random.rand(self.dim)
    	while self.flag:
    	    itr += 1
	    self.flag = False
	    self.update()
	    '''
    	    print 'After iteration {} weights are:'.format(itr)
    	    print self.weights
    	    print '-'*80 + '\n'
	    '''
	print 'Converged after {} iterations'.format(itr)	

    def predict(self, test_set):
	"""predict test set labels"""
	test_set = self.augmented_set(test_set)
	pred_labels_ = (np.dot(test_set, self.weights) > 0.0).astype(int)
	idx = pred_labels_==0
	pred_labels_[idx] = self.labels_[0]
	pred_labels_[~idx] = self.labels_[1]
	return list(pred_labels_)

    def plot_boundary(self):
	"""find two points on the seperation line defined by the final 
	weight vector and join these points to plot seperating line"""
    	a, b, c = self.weights
	m = -a/b # slope of line
	X = self.train_set.copy()
	X[self.idx] = -X[self.idx]
	x_min, y_min = np.min(X[:,0])-1, np.min(X[:,1])-1
	x_max, y_max = np.max(X[:,1])+1, np.max(X[:,1])+1

	# plot data-points
	pl.scatter(X[self.idx][:,0], X[self.idx][:,1],\
		    c='r', edgecolor='r')
	pl.scatter(X[~self.idx][:,0], X[~self.idx][:,1],\
		    c='b', edgecolor='b')

	# plot seperation line
	# equation of seperation line is given by
	# ax + by + c = 0
	# at x1 = 0, y1 = -c/b, at y2 = 0, x2 = -c/a
	if m < 0: # negative slope
	   pl.plot([x_min, x_max], [-(c+a*x_min)/b, -(c+a*x_max)/b], '--k')
	else: # positive slope
	   pl.plot([-(c+b*y_min)/a, -(c+b*y_max)/a], [y_min, y_max], '--k')
	pl.xlim(x_min, x_max)
	pl.ylim(y_min, y_max)
	pl.show()
