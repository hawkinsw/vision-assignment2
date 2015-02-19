#!/usr/bin/python

import numpy
import math
import matplotlib.pyplot as plt
import skimage
import skimage.io

def g(observation, estimate):
	exponent = -1.0*numpy.dot(estimate.T, observation)
	return 1.0/(1.0 + numpy.exp(numpy.add.reduce(numpy.add.reduce(exponent))))

def FitLogReg(observations, classifications, mu):
	assert len(observations) == len(classifications),\
		"Number of observations and classifications must match."
	_w = numpy.zeros((observations[0].shape[0]+1,1))

	for i in range(len(observations)):
		print("obs shape: %s" % str(observations[i].shape))
		observations[i] = numpy.vstack([observations[i], [1]])
		print("obs shape: %s" % str(observations[i].shape))

	iterations = 0
	while True:
		summation = numpy.zeros(_w.shape)
		for o,c in zip(observations, classifications):
			summation += (c - g(o,_w))*o
		_w = _w + mu*summation
		iterations += 1
		if iterations > 100:
			break
	return _w

class LogReg:
	def __init__(self, estimate):
		self.estimate = estimate
	def evaluate(self, observation):
		return g(numpy.vstack([observation, [1]]), self.estimate)
	def __repr__(self):
		return repr(self.estimate)
	def imsave(self, filename):
		skimage.io.imsave(filename, numpy.reshape(self.estimate[0:144,:], (12,12)))
