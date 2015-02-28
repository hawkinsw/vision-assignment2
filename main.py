#!/usr/bin/python

from dataprep import dataprep
import gaussian
import logreg
from operator import itemgetter

import random
import skimage
import skimage.io
import skimage.draw
import skimage.color
import numpy
# We did this once. We can do it again, if we have to.
# Until then, keep it commented out for reproducibility.

image_column_shape = (144,1)

class Evaluator:
	def __init__(self, obj):
		self.obj = obj
	def evaluate(self, candidate):
		return self.obj.evaluate(candidate)

class GaussianEvaluator:
	def __init__(self, face, not_face):
		self.face = face
		self.not_face = not_face
	def evaluate(self, candidate):
		face_e = self.face.evaluate(candidate)
		random_e = self.not_face.evaluate(candidate)
		return (face_e>random_e, face_e-random_e)

class LinearEvaluator:
	def __init__(self, logreg):
		self.logreg = logreg
	def evaluate(self, candidate):
		value = self.logreg.evaluate(candidate)
		return (value > 0.5, value)

def __initialize_data__():
	d = dataprep.GroundTruth("./dataprep/data/",\
		filename="./dataprep/data-hires.html",\
		expansion=0.20)
	d.draw_face_squares("./output/")
	patches = d.extract_face_patches()
	dataprep.generate_mosaic(patches, "./actual/face-mosaic.gif")
	patches = d.extract_random_patches((12,12), count=120)
	dataprep.generate_mosaic(patches, "./actual/random-mosaic.gif")

def find_faces(input_image_filename,
	output_image_filename,
	evaluator,
	neighborhood = (12,12)):

	test_image = skimage.color.rgb2gray(
		skimage.img_as_float(skimage.io.imread(input_image_filename)))
	test_image_height, test_image_width = test_image.shape
	found_faces = numpy.zeros(test_image.shape)

	for x in range(test_image_width-12):
		for y in range(test_image_height-12):
			test_patch = test_image[y:y+12,x:x+12]
			#print("test patch shape: %s" % str(test_patch.shape))
			test_patch = numpy.reshape(test_patch, image_column_shape)
			is_face, value = evaluator.evaluate(test_patch)
			if is_face:
				found_faces[y,x] = value

	#found_faces = sorted(found_faces, key=itemgetter(0))
	for x in range(found_faces.shape[1]):
		for y in range(found_faces.shape[0]):
			if found_faces[y,x] == 0.0:
				continue
			#
			# non-maximum suppression.
			#
			is_max = True
			for xx in range(-1*(neighborhood[1]/2), neighborhood[1]/2):
				for yy in range(-1*(neighborhood[0]/2), neighborhood[0]/2):
					if y+yy > 0 and y+yy < found_faces.shape[0] and\
					   x+xx > 0 and x+xx < found_faces.shape[1] and\
						 found_faces[y+yy,x+xx] > found_faces[y,x]:
						is_max = False
						break
			if not is_max:
				continue
			rr, cc = skimage.draw.line(y, x, y, x+12)
			test_image[rr,cc] = 1.0
			rr, cc = skimage.draw.line(y, x+12, y+12, x+12)
			test_image[rr,cc] = 1.0
			rr, cc = skimage.draw.line(y+12, x+12, y+12, x)
			test_image[rr,cc] = 1.0
			rr, cc = skimage.draw.line(y+12, x, y, x)
			test_image[rr,cc] = 1.0
	skimage.io.imsave(output_image_filename, test_image)

def build_linear_classifier_evaluator():
	#
	# First 120 patches are going to be faces.
	# Second 120 patches are not.
	#
	classifications = numpy.zeros((240,1))
	classifications[0:119] = 1.0
	classifications[120:239] = 0.0

	#
	# Preallocate 240 spots for image patches.
	#
	observations = []
	observations_counter = 0
	[observations.append(0) for i in range(240)]

	#
	# Prepare the training data for face patches.
	#
	face_patches = dataprep.read_mosaic("./actual/face-mosaic.gif")
	face_reshaped = [numpy.reshape(p, image_column_shape) for p in face_patches]
	for r in face_reshaped:
		observations[observations_counter] = r
		observations_counter+=1

	#
	# Prepare the training data for random patches.
	#
	random_patches = dataprep.read_mosaic("./actual/random-mosaic.gif")
	random_reshaped = [numpy.reshape(p,image_column_shape)for p in random_patches]
	for r in random_reshaped:
		observations[observations_counter] = r
		observations_counter+=1

	print("# of classifications: " + str(classifications.size))
	print("# of observations: " + str(len(observations)))

	l = logreg.LogReg(logreg.FitLogReg(observations, classifications, 0.01))

	return Evaluator(LinearEvaluator(l))

def build_gaussian_evaluator():
	#
	# Prepare the training data for face patches.
	#
	face_patches = dataprep.read_mosaic("./actual/face-mosaic.gif")
	reshaped_face = [numpy.reshape(p, image_column_shape) for p in face_patches]
	face_g = gaussian.FitGaussian(image_column_shape, reshaped_face)

	#
	# Prepare the training data for random patches.
	#
	random_patches = dataprep.read_mosaic("./actual/random-mosaic.gif")
	reshaped_random=[numpy.reshape(p, image_column_shape) for p in random_patches]
	random_g = gaussian.FitGaussian(image_column_shape, reshaped_random)

	return Evaluator(GaussianEvaluator(face_g, random_g))

if __name__=='__main__':
	#__initialize_data__()
	linear_evaluator = build_linear_classifier_evaluator()
	gaussian_evaluator = build_gaussian_evaluator()

	evaluator_tags = [(linear_evaluator, "linear"),
		(gaussian_evaluator, "gaussian")]
	for evaluator, tag in evaluator_tags:
		find_faces("./test_input/solidbg.jpg",
			"./test_output/found_solidbg_" + tag + ".jpg",
			evaluator)
		find_faces("./test_input/randombg.jpg",
			"./test_output/found_randombg_" + tag + ".jpg",
			evaluator)
		find_faces("./test_input/solidbg-bioid.jpg",
			"./test_output/found_solidbg-bioid_" + tag + ".jpg",
			evaluator)
		find_faces("./test_input/randombg-bioid.jpg",
			"./test_output/found_randombg-bioid_" + tag + ".jpg",
			evaluator)
