#!/usr/bin/python

from dataprep import dataprep
import gaussian

import random
import skimage
import skimage.io
import numpy

if __name__=='__main__':
	#
	# We did this once. We can do it again, if we have to.
	# Until then, keep it commented out for reproducibility.
	#
	#d = dataprep.GroundTruth("./dataprep/data/", filename="./dataprep/data.html")
	#patches = d.extract_face_patches()
	#dataprep.generate_mosaic(patches, "./actual/face-mosaic.gif")
	#patches = d.extract_random_patches((12,12), count=120)
	#dataprep.generate_mosaic(patches, "./actual/random-mosaic.gif")

	image_column_shape = (144,1)
	patches = dataprep.read_mosaic("./actual/face-mosaic.gif")
	reshaped = [numpy.reshape(p, image_column_shape) for p in patches]
	image_g = gaussian.FitGaussian(image_column_shape, reshaped)

	image_column_shape = (144,1)
	patches = dataprep.read_mosaic("./actual/random-mosaic.gif")
	reshaped = [numpy.reshape(p, image_column_shape) for p in patches]
	random_g = gaussian.FitGaussian(image_column_shape, reshaped)

	find_image = skimage.img_as_float(skimage.io.imread("dataprep/data/newtest/er.gif"))
	find_image_height, find_image_width = find_image.shape
	counter = 0
	for y in range(find_image_height-20):
		for x in range(find_image_width-20):
			test_patch = find_image[y:y+20,x:x+20]
			test_patch = skimage.transform.resize(test_patch, (12,12))
			image_e = image_g.evaluate(numpy.reshape(test_patch, (144,1)))
			random_e = random_g.evaluate(numpy.reshape(test_patch, (144,1)))
			if (image_e > random_e):
				print("image_e: %f random_e: %f" % (image_e, random_e))
