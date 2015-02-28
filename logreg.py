#!/usr/bin/python

import numpy
import math
import matplotlib.pyplot as plt
import skimage
import skimage.io

class Debug:
	@classmethod
	def Print(cls, str):
		print(str)
		pass

def g(observation, estimate):
	#
	# Use ravel() here to make these vectors
	# instead of column/row matrices.
	#
	exponent = -1.0*(numpy.dot(estimate.ravel(),observation.ravel()))
	return 1.0/(1.0 + numpy.exp(exponent))

def FitLogReg(observations, classifications, mu,
	convergence=1.0e-2, max_iterations=300):
	assert len(observations) == len(classifications),\
		"Number of observations and classifications must match."

	error_csv = open("./logreg.csv", 'w')
	_w = numpy.zeros((observations[0].shape[0]+1,1))

	#
	# "Canonicalize" the data representations.
	#
	for i in range(len(observations)):
		observations[i] = numpy.vstack([observations[i], [1]])

	iterations = 0
	while True:
		summation = numpy.zeros(_w.shape)
		for o,c in zip(observations, classifications):
			summation += (c - g(o,_w))*o

		scaled_summation = mu*summation
		_w_delta_sum = numpy.add.reduce(scaled_summation)
		_w = _w + scaled_summation

		if math.fabs(_w_delta_sum) <= convergence:
			break
		iterations += 1
		error_csv.write("%d,%f\n" % (iterations, _w_delta_sum))
		if iterations > max_iterations:
			break

	error_csv.close()
	return _w

class LogReg:
	def __init__(self, estimate):
		self.estimate = estimate
	def evaluate(self, observation):
		return g(numpy.vstack([observation, [1]]), self.estimate)
	def __repr__(self):
		return repr(self.estimate)
