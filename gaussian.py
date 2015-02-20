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
	result = A.T.dot(A)/(1.0*rows)
	"""
	print("A.T.shape: %s" % str(A.T.shape))
	print("A.shape: %s" % str(A.shape))
	print("result.shape: %s" % str(result.shape))
	"""
	return result

def FitGaussian(sample_shape, samples):
	_mu = mu(sample_shape, samples)
	_sigma = sigma(sample_shape, _mu, samples)
	return Gaussian(_mu, _sigma)

class Gaussian:
	def __init__(self, mu, sigma, tau=1.0e-0):
		self.mu = mu
		(u,s,u_t) = numpy.linalg.svd(sigma)

		"""
		print("u.shape: %s" % str(u.shape))
		print("s.shape: %s" % str(numpy.diag(s).shape))
		print("u_t.shape: %s" % str(u_t.shape))
		print("s: " + str(s))

		"""
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

		self.sigma = trunc_u.dot(numpy.diag(trunc_s)).dot(trunc_u_t)

		"""
		print("trunc_u.shape: %s" % str(trunc_u.shape))
		print("trunc_s.shape: %s" % str(numpy.diag(trunc_s).shape))
		print("trunc_u_t.shape: %s" % str(trunc_u_t.shape))
		print("self.sigma.shape: %s" % str(self.sigma.shape))
		"""
		# http://en.wikipedia.org/wiki/Singular_value_decomposition#Applications_of_the_SVD
		self.inv_sigma = trunc_u.dot(numpy.diag(inv_trunc_s)).dot(trunc_u_t)
		self.det_sigma = numpy.multiply.reduce(trunc_s)

	def evaluate(self, x):
		denominator = (numpy.sqrt(math.pow(2.0*numpy.pi, self.k)*self.det_sigma))
		exponent=-0.5*numpy.transpose(x-self.mu).dot(self.inv_sigma).dot(x-self.mu)
		"""
		print("exponent.shape: %s" % str(exponent.shape))
		"""
		exponent = numpy.add.reduce(numpy.add.reduce(exponent))
		return numpy.exp(exponent)/denominator
