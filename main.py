#!/usr/bin/python

from dataprep import dataprep
import gaussian
import logreg

import random
import skimage
import skimage.io
import numpy
# We did this once. We can do it again, if we have to.
# Until then, keep it commented out for reproducibility.

def __initialize_data__():
	d = dataprep.GroundTruth("./dataprep/data/",\
		filename="./dataprep/data.html",\
		expansion=0.30)
	#d.draw_face_squares("./output2/")
	patches = d.extract_face_patches()
	dataprep.generate_mosaic(patches, "./actual/face-mosaic.gif")
	patches = d.extract_random_patches((12,12), count=120)
	dataprep.generate_mosaic(patches, "./actual/random-mosaic.gif")

def linear_classifier_detector():
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
	image_column_shape = (144,1)
	face_patches = dataprep.read_mosaic("./actual/face-mosaic.gif")
	face_reshaped = [numpy.reshape(p, image_column_shape) for p in face_patches]
	for r in face_reshaped:
		observations[observations_counter] = r
		observations_counter+=1

	#
	# Find a random face so that we can test!
	#
	random_face_index = random.randint(0, len(face_patches)-1)
	random_face = face_patches[random_face_index]
	random_face = numpy.reshape(random_face, image_column_shape)

	#
	# Prepare the training data for random patches.
	#
	random_patches = dataprep.read_mosaic("./actual/random-mosaic.gif")
	random_reshaped = [numpy.reshape(p,image_column_shape)for p in random_patches]
	for r in random_reshaped:
		observations[observations_counter] = r
		observations_counter+=1

	#
	# Find a random random so that we can test!
	#
	random_random_index = random.randint(0, len(random_patches)-1)
	random_random = random_patches[random_random_index]
	random_random = numpy.reshape(random_random, image_column_shape)

	print("# of classifications: " + str(classifications.size))
	print("# of observations: " + str(len(observations)))

	l = logreg.LogReg(logreg.FitLogReg(observations, classifications, 0.01))
	#print("w: " + str(l))

	print("random face index: " + str(random_face_index))
	print("random face evaluation: " + str(l.evaluate(random_face)))
	print("random random index: " + str(random_random_index))
	print("random random evaluation: " + str(l.evaluate(random_random)))

def gaussian_detector():
	image_column_shape = (144,1)

	#
	# Prepare the training data for face patches.
	#
	face_patches = dataprep.read_mosaic("./actual/face-mosaic.gif")
	reshaped_face = [numpy.reshape(p, image_column_shape) for p in face_patches]
	face_g = gaussian.FitGaussian(image_column_shape, reshaped_face)

	#
	# Find a random face so that we can test!
	#
	random_face = face_patches[random.randint(0, len(face_patches)-1)]
	random_face = numpy.reshape(random_face, image_column_shape)

	#
	# Prepare the training data for random patches.
	#
	random_patches = dataprep.read_mosaic("./actual/random-mosaic.gif")
	reshaped_random=[numpy.reshape(p, image_column_shape) for p in random_patches]
	random_g = gaussian.FitGaussian(image_column_shape, reshaped_random)

	#
	# Find a random random so that we can test!
	#
	random_random = random_patches[random.randint(0, len(random_patches)-1)]
	random_random = numpy.reshape(random_random, image_column_shape)

	face_e = face_g.evaluate(random_face)
	random_e = random_g.evaluate(random_face)
	print("Face:")
	print("face e: " + repr(face_e))
	print("random e: " + repr(random_e))
	if (face_e>random_e):
		print("Face detected.")
	else:
		print("Face NOT detected.")

	face_e = face_g.evaluate(random_random)
	random_e = random_g.evaluate(random_random)
	print("Random:")
	print("face e: " + repr(face_e))
	print("random e: " + repr(random_e))
	if (face_e>random_e):
		print("Face detected.")
	else:
		print("Face NOT detected.")

if __name__=='__main__':
	#linear_classifier_detector()
	gaussian_detector()
