#!/usr/bin/python

from dataprep import dataprep
import gaussian
import logreg
from operator import itemgetter

import math
import random
import skimage
import skimage.io
import skimage.draw
import skimage.color
import numpy

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
	patches = d.extract_face_patches(count=100)
	dataprep.generate_mosaic(patches, "./actual/face-mosaic.gif", mosaic_width=10)
	patches = d.extract_random_patches((12,12), count=100)
	dataprep.generate_mosaic(patches, "./actual/random-mosaic.gif", mosaic_width=10)

def list_faces(input_image,
	evaluator):
	face_list = []
	input_image_width, input_image_height = input_image.shape
	for x in range(input_image_width-12):
		for y in range(input_image_height-12):
			test_patch = input_image[y:y+12,x:x+12]
			#print("test patch shape: %s" % str(test_patch.shape))
			test_patch = numpy.reshape(test_patch, image_column_shape)
			is_face, value = evaluator.evaluate(test_patch)
			if is_face:
				face_list.append(((y,x), value))
	return face_list
def build_linear_classifier_evaluator():
	#
	# First 120 patches are going to be faces.
	# Second 120 patches are not.
	#
	classifications = numpy.zeros((200,1))
	classifications[0:99] = 1.0
	classifications[100:199] = 0.0

	#
	# Preallocate 240 spots for image patches.
	#
	observations = []
	observations_counter = 0
	[observations.append(0) for i in range(200)]

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

	l = logreg.LogReg(logreg.FitLogReg(observations, classifications, 0.001, convergence=0.00001))

	return Evaluator(LinearEvaluator(l))

def build_gaussian_evaluator():
	#
	# Prepare the training data for face patches.
	#
	face_patches = dataprep.read_mosaic("./actual/face-mosaic.gif")
	reshaped_face = [numpy.reshape(p, image_column_shape) for p in face_patches]
	face_g = gaussian.FitGaussian(image_column_shape, reshaped_face, name="face")
	face_g.save_mu("./actual/face-average.gif")

	#
	# Prepare the training data for random patches.
	#
	random_patches = dataprep.read_mosaic("./actual/random-mosaic.gif")
	reshaped_random=[numpy.reshape(p, image_column_shape) for p in random_patches]
	random_g = gaussian.FitGaussian(image_column_shape, reshaped_random, name="random")
	random_g.save_mu("./actual/random-average.gif")

	return Evaluator(GaussianEvaluator(face_g, random_g))

def self_eating_test():
	linear_evaluator = build_linear_classifier_evaluator()
	gaussian_evaluator = build_gaussian_evaluator()
	evaluator_tags = [(linear_evaluator, "linear"),
		(gaussian_evaluator, "gaussian")]
	mosaic_tags = [("./actual/face-mosaic.gif", "Faces"),
		("./actual/random-mosaic.gif", "Random")]
	for evaluator, evaluator_tag in evaluator_tags:
		for (test_set_filename, test_set_tag) in mosaic_tags:
			face_count = 0
			non_face_count = 0
			test_patches = dataprep.read_mosaic(test_set_filename)
			test_patches=[numpy.reshape(p, image_column_shape) for p in test_patches]
			for patch in test_patches:
				(is_face, value) = evaluator.evaluate(patch)
				if is_face:
					face_count += 1
				else:
					non_face_count += 1
			print("%s on %s:" % (evaluator_tag, test_set_tag))
			print("face_count: %d, non_face_count: %d" % (face_count, non_face_count))

def test_basic_find_faces_mosaic():
	linear_evaluator = build_linear_classifier_evaluator()
	gaussian_evaluator = build_gaussian_evaluator()

	evaluator_tags = [(linear_evaluator, "linear"),
		(gaussian_evaluator, "gaussian")]
	for evaluator, tag in evaluator_tags:
		basic_find_faces("./test_input/face_mosaic_actual.gif",
			"./test_output/nnm_found_face_mosaic_actual_" + tag + ".gif",
			evaluator, neighborhood=None)
		basic_find_faces("./test_input/random_mosaic_actual.gif",
			"./test_output/nnm_found_random_mosaic_actual_" + tag + ".gif",
			evaluator, neighborhood=None)
		basic_find_faces("./test_input/face_mosaic_actual.gif",
			"./test_output/found_face_mosaic_actual_" + tag + ".gif",
			evaluator)
		basic_find_faces("./test_input/random_mosaic_actual.gif",
			"./test_output/found_random_mosaic_actual_" + tag + ".gif",
			evaluator)

		basic_find_faces("./test_input/face_mosaic.gif",
			"./test_output/nnm_found_face_mosaic_" + tag + ".gif",
			evaluator, neighborhood=None)
		basic_find_faces("./test_input/random_mosaic.gif",
			"./test_output/nnm_found_random_mosaic_" + tag + ".gif",
			evaluator, neighborhood=None)
		basic_find_faces("./test_input/face_mosaic.gif",
			"./test_output/found_face_mosaic_" + tag + ".gif",
			evaluator)
		basic_find_faces("./test_input/random_mosaic.gif",
			"./test_output/found_random_mosaic_" + tag + ".gif",
			evaluator)

def test_basic_find_faces():
	linear_evaluator = build_linear_classifier_evaluator()
	gaussian_evaluator = build_gaussian_evaluator()

	evaluator_tags = [(linear_evaluator, "linear"),
		(gaussian_evaluator, "gaussian")]
	for evaluator, tag in evaluator_tags:
		#
		# w/o non maximal suppression.
		#
		basic_find_faces("./test_input/solidbg.gif",
			"./test_output/nnm_found_solidbg_" + tag + ".gif",
			evaluator, neighborhood=None)
		basic_find_faces("./test_input/solidbg2.gif",
			"./test_output/nnm_found_solidbg2_" + tag + ".gif",
			evaluator, neighborhood=None)
		basic_find_faces("./test_input/randombg.gif",
			"./test_output/nnm_found_randombg_" + tag + ".gif",
			evaluator, neighborhood=None)
		basic_find_faces("./test_input/randombg2.gif",
			"./test_output/nnm_found_randombg2_" + tag + ".gif",
			evaluator, neighborhood=None)
		basic_find_faces("./test_input/randombg6.gif",
			"./test_output/nnm_found_randombg6_" + tag + ".gif",
			evaluator, neighborhood=None)
		basic_find_faces("./test_input/solidbg6.gif",
			"./test_output/nnm_found_solidbg6_" + tag + ".gif",
			evaluator, neighborhood=None)
		basic_find_faces("./test_input/solidbg-bioid.gif",
			"./test_output/nnm_found_solidbg-bioid_" + tag + ".gif",
			evaluator, neighborhood=None)
		basic_find_faces("./test_input/randombg-bioid.gif",
			"./test_output/nnm_found_randombg-bioid_" + tag + ".gif",
			evaluator, neighborhood=None)
		#
		# w/ non-maximal suppression
		#
		basic_find_faces("./test_input/solidbg.gif",
			"./test_output/found_solidbg_" + tag + ".gif",
			evaluator)
		basic_find_faces("./test_input/solidbg2.gif",
			"./test_output/found_solidbg2_" + tag + ".gif",
			evaluator)
		basic_find_faces("./test_input/randombg.gif",
			"./test_output/found_randombg_" + tag + ".gif",
			evaluator)
		basic_find_faces("./test_input/randombg2.gif",
			"./test_output/found_randombg2_" + tag + ".gif",
			evaluator)
		basic_find_faces("./test_input/randombg6.gif",
			"./test_output/found_randombg6_" + tag + ".gif",
			evaluator)
		basic_find_faces("./test_input/solidbg6.gif",
			"./test_output/found_solidbg6_" + tag + ".gif",
			evaluator)
		basic_find_faces("./test_input/solidbg-bioid.gif",
			"./test_output/found_solidbg-bioid_" + tag + ".gif",
			evaluator)
		basic_find_faces("./test_input/randombg-bioid.gif",
			"./test_output/found_randombg-bioid_" + tag + ".gif",
			evaluator)
def test_pyramid_find_faces():
	linear_evaluator = build_linear_classifier_evaluator()
	gaussian_evaluator = build_gaussian_evaluator()

	evaluator_tags = [(linear_evaluator, "linear"),
		(gaussian_evaluator, "gaussian")]
	for evaluator, tag in evaluator_tags:
		#
		# w/o non maximal suppression.
		#
		"""
		pyramid_find_faces("./test_input/solidbg-different-sizes.gif",
			"./test_output/nnm_found_solidbg-different-sizes_pyramid_"+tag+".gif",
			evaluator, neighborhood=None)
		pyramid_find_faces("./test_input/solidbg-different-many-sizes.gif",
			"./test_output/nnm_found_solidbg-different-many-sizes_pyramid_"+tag+".gif",
			evaluator, neighborhood=None)
		"""
		pyramid_find_faces("./test_input/randombg-different-sizes.gif",
			"./test_output/nnm_found_randombg-different-sizes_pyramid_"+tag+".gif",
			evaluator, neighborhood=None)
		"""
		pyramid_find_faces("./test_input/solidbg-different-many-sizes-bioid.gif",
			"./test_output/nnm_found_solidbg-different-many-sizes-bioid_pyramid_"+tag+".gif",
			evaluator, neighborhood=None)
		pyramid_find_faces("./test_input/solidbg-different-sizes2.gif",
			"./test_output/nnm_found_solidbg-different-sizes2_pyramid_"+tag+".gif",
			evaluator, neighborhood=None)
		#
		# w/ non-maximal suppression
		#
		pyramid_find_faces("./test_input/solidbg-different-sizes.gif",
			"./test_output/found_solidbg-different-sizes_pyramid_"+tag+".gif",
			evaluator)
		pyramid_find_faces("./test_input/randombg-different-sizes.gif",
			"./test_output/found_randombg-different-sizes_pyramid_"+tag+".gif",
			evaluator)
		pyramid_find_faces("./test_input/solidbg-different-many-sizes.gif",
			"./test_output/found_solidbg-different-many-sizes_pyramid_"+tag+".gif",
			evaluator)
		pyramid_find_faces("./test_input/solidbg-different-many-sizes-bioid.gif",
			"./test_output/found_solidbg-different-many-sizes-bioid_pyramid_"+tag+".gif",
			evaluator)
		pyramid_find_faces("./test_input/solidbg-different-sizes2.gif",
			"./test_output/found_solidbg-different-sizes2_pyramid_"+tag+".gif",
			evaluator)
		"""
def basic_find_faces(input_image_filename,
	output_image_filename,
	evaluator,
	neighborhood = (12,12)):

	test_image = skimage.color.rgb2gray(
		skimage.img_as_float(skimage.io.imread(input_image_filename)))
	test_image_height, test_image_width = test_image.shape
	found_faces = numpy.zeros(test_image.shape)

	face_list = list_faces(test_image, evaluator)

	for (y,x), value in face_list:
		found_faces[y,x] = value

	for x in range(found_faces.shape[1]):
		for y in range(found_faces.shape[0]):
			if found_faces[y,x] == 0:
				continue
			#
			# non-maximum suppression.
			#
			is_max = True
			if neighborhood != None:
				for xx in range(-1*(neighborhood[1]/2), neighborhood[1]/2):
					for yy in range(-1*(neighborhood[0]/2), neighborhood[0]/2):
						if xx ==0 and yy == 0: continue
						if y+yy > 0 and y+yy < found_faces.shape[0] and\
						   x+xx > 0 and x+xx < found_faces.shape[1] and\
							 found_faces[y+yy,x+xx] > found_faces[y,x]:
							is_max = False
							break
					if not is_max:
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

def pyramid_find_faces(input_image_filename,
	output_image_filename,
	evaluator,
	neighborhood = (12,12),
	scale=1.1,
	octaves = 25):

	original_test_image = skimage.color.rgb2gray(
		skimage.img_as_float(skimage.io.imread(input_image_filename)))
	output_image = numpy.zeros(original_test_image.shape, dtype=object)
	test_image = original_test_image

	o = 0
	for o in range(octaves):
		print("Starting octave %d." % o)
		print("Test image shape: %s." % str(test_image.shape))

		#
		# When the image is small enough!
		#
		if test_image.shape[0] == 1 or test_image.shape[1] == 1:
			break

		skimage.io.imsave("./output/test_image-" + str(o) + ".gif", test_image)

		# find faces
		found_faces = list_faces(test_image, evaluator)
		found_test_image = numpy.zeros(test_image.shape)
		for (y,x), value in found_faces:
			found_test_image[y,x] = value#*(math.pow(scale,o))

		#
		# Non max suppression per level!
		for x in range(found_test_image.shape[1]):
			for y in range(found_test_image.shape[0]):
				if found_test_image[y,x] == 0:
					continue
				is_max = True
				value = found_test_image[y,x]
				if neighborhood != None:
					for xx in range(-1*(neighborhood[1]/2), neighborhood[1]/2):
						for yy in range(-1*(neighborhood[0]/2), neighborhood[0]/2):
							if y+yy > 0 and y+yy < found_test_image.shape[0] and\
							   x+xx > 0 and x+xx < found_test_image.shape[1] and\
								 found_test_image[y+yy,x+xx] > value:
									is_max = False
									break
						if not is_max:
							break
				if not is_max:
					continue
				# convert found face locations back to original
				original_y = y*(math.pow(scale,o))
				original_x = x*(math.pow(scale,o))
				print("Found face at (y,x):(%d,%d) -> (original_y,original_x):(%d,%d)"%
					(y,x,original_y,original_x))
				output_image[original_y,original_x] = (value,
					(12*(math.pow(scale,o)),12*(math.pow(scale,o))))
		else:
			if len(found_faces) == 0:
				print("Found no faces at %d." % o)
		#
		# Prepare for the next level
		#
		# 1. blur
		test_image = skimage.filter.gaussian_filter(test_image,
			2*scale/6.0, # we all know where this came from!
			mode='wrap')
		# 2. subsample
		test_image = skimage.transform.rescale(test_image, 1.0/scale)

	# 
	# Draw boxes around found faces.
	#
	for x in range(output_image.shape[1]):
		for y in range(output_image.shape[0]):
			if output_image[y,x] == 0:
				continue
			found_face_width = output_image[y,x][1][1]
			found_face_height = output_image[y,x][1][0]
			#
			# Do a "global" non-maximal suppression with 
			# a neighborhood that is proportional to the
			# size of the detected face.
			#
			is_max = True
			if neighborhood != None:
				for xx in range(int(-1*(found_face_width/2)), int(found_face_width/2)):
					for yy in range(int(-1*(found_face_height/2)), int(found_face_height/2)):
						if y+yy > 0 and y+yy < output_image.shape[0] and\
						   x+xx > 0 and x+xx < output_image.shape[1] and\
							 output_image[y+yy,x+xx] != 0 and\
							 output_image[y+yy,x+xx][0] > output_image[y,x][0]:
								is_max = False
								break
					if not is_max:
						break
			if not is_max:
				continue
			right =math.floor(min(x+found_face_width, original_test_image.shape[1]-1))
			bottom=math.floor(min(y+found_face_height,original_test_image.shape[0]-1))
			right = int(right)
			bottom = int(bottom)

			rr, cc = skimage.draw.line(y, x, y, right)
			original_test_image[rr,cc] = 1.0
			rr, cc = skimage.draw.line(y, right, bottom, right)
			original_test_image[rr,cc] = 1.0
			rr, cc = skimage.draw.line(bottom, right, bottom, x)
			original_test_image[rr,cc] = 1.0
			rr, cc = skimage.draw.line(bottom, x, y, x)
			original_test_image[rr,cc] = 1.0

	skimage.io.imsave(output_image_filename, original_test_image)

if __name__=='__main__':
	#__initialize_data__()
	#self_eating_test()
	#test_basic_find_faces()
	#test_basic_find_faces_mosaic()
	#test_pyramid_find_faces()
	pass
