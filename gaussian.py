#!/usr/bin/python

import numpy
import math
import matplotlib.pyplot as plt

def mu(shape, arrays):
	result = numpy.zeros(shape)
	for a in arrays:
		result += a
	return result/(1.0*result.size)

def sigma(shape, mu, arrays):
	rows = len(arrays)
	columns = shape[0]
	A = numpy.zeros((rows,columns))
	row = 0
	for a in arrays:
		A[row] = numpy.transpose(a-mu)
		row += 1
	return numpy.dot(A.T,A)/(1.0*rows)

def FitGaussian(sample_shape, samples):
	_mu = mu(sample_shape, samples)
	_sigma = sigma(sample_shape, _mu, samples)
	return Gaussian(_mu, _sigma)

class Gaussian:
	def __init__(self, mu, sigma, tau=1.0e-0):
		self.mu = mu
		(u,s,u_t) = numpy.linalg.svd(sigma)
		#
		# Find out how much of sigma is larger than tau.
		#
		self.k = s[s>tau].size

		#
		# Truncate to those values.
		#
		trunc_u = u[:,0:self.k]
		trunc_s = s[0:self.k]
		trunc_u_t = trunc_u.T
		inv_trunc_s = 1.0/trunc_s
		self.sigma = numpy.dot(numpy.dot(trunc_u, numpy.diag(trunc_s)), trunc_u_t)
		# http://en.wikipedia.org/wiki/Singular_value_decomposition#Applications_of_the_SVD
		self.inv_sigma = numpy.dot(numpy.dot(trunc_u, numpy.diag(inv_trunc_s)), trunc_u_t)
		self.det_sigma = reduce(lambda x,y: x*y, trunc_s)
		print("det_sigma: " + str(self.det_sigma))


	def evaluate(self, x):
		denominator = 1.0 /\
			(numpy.sqrt(math.pow(2.0*numpy.pi, self.k)*self.det_sigma))
		exponent = -0.5*numpy.dot(numpy.dot(numpy.transpose(x-self.mu),self.inv_sigma),(x-self.mu))
		return reduce(lambda x,y: x+y, (numpy.exp(exponent)*denominator).ravel().tolist())
