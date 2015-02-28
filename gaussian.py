#!/usr/bin/python

import numpy
import math
import matplotlib.pyplot as plt

class Debug:
	@classmethod
	def Print(cls, str):
		#print(str)
		pass

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
	Debug.Print("A.T.shape: %s" % str(A.T.shape))
	Debug.Print("A.shape: %s" % str(A.shape))
	Debug.Print("result.shape: %s" % str(result.shape))
	return result

def FitGaussian(sample_shape, samples):
	_mu = mu(sample_shape, samples)
	_sigma = sigma(sample_shape, _mu, samples)
	return Gaussian(_mu, _sigma)

class Gaussian:
	def __init__(self, mu, sigma, tau=1.0e-5):
		self.mu = mu
		(u,s,u_t) = numpy.linalg.svd(sigma)

		Debug.Print("u.shape: %s" % str(u.shape))
		Debug.Print("s.shape: %s" % str(numpy.diag(s).shape))
		Debug.Print("u_t.shape: %s" % str(u_t.shape))
		Debug.Print("s: " + str(s))

		#
		# Write the singular values to a file
		# so that we can graph it! Open with
		# truncation so that we always get
		# these values.
		#
		f = open("./singular.csv", 'w')
		s_list = numpy.ndarray.tolist(s)
		s_string = "\n".join([str(ss) for ss in s_list])
		f.write(s_string)
		f.close()

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

		Debug.Print("trunc_u.shape: %s" % str(trunc_u.shape))
		Debug.Print("trunc_s.shape: %s" % str(numpy.diag(trunc_s).shape))
		Debug.Print("trunc_u_t.shape: %s" % str(trunc_u_t.shape))
		Debug.Print("self.sigma.shape: %s" % str(self.sigma.shape))
		# http://en.wikipedia.org/wiki/Singular_value_decomposition#Applications_of_the_SVD
		self.inv_sigma = trunc_u.dot(numpy.diag(inv_trunc_s)).dot(trunc_u_t)
		self.det_sigma = numpy.multiply.reduce(trunc_s)

	def evaluate(self, x):
		denominator = (numpy.sqrt(math.pow(2.0*numpy.pi, self.k)*self.det_sigma))
		exponent=-0.5*numpy.transpose(x-self.mu).dot(self.inv_sigma).dot(x-self.mu)
		Debug.Print("exponent.shape: %s" % str(exponent.shape))
		exponent = numpy.add.reduce(numpy.add.reduce(exponent))
		return numpy.exp(exponent)/denominator
