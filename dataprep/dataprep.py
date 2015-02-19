#!/usr/bin/python

from __future__ import print_function
from operator import itemgetter
import os.path

import urllib2
from bs4 import BeautifulSoup
import skimage
import skimage.io
import skimage.draw
import skimage.transform
import re
import numpy
import random

def unit_test_intersect():
	s = (5,5,10,10)
	ss = [(7,7,12,12,)]
	assert intersect_any(s, ss), "%s and %s should intersect!" % (s,ss)
	s = (5,5,10,10)
	ss = [(12,12,14,14,)]
	assert not intersect_any(s, ss), "%s and %s should intersect!" % (s,ss)
	s = (5,5,10,10)
	ss = [(2,7,12,12)]
	assert intersect_any(s, ss), "%s and %s should intersect!" % (s,ss)
	s = (5,5,10,10)
	ss = [(2,2,7,7)]
	assert intersect_any(s, ss), "%s and %s should intersect!" % (s,ss)
	s = (2,2,7,7)
	ss = [(5,5,10,10)]
	assert intersect_any(s, ss), "%s and %s should intersect!" % (s,ss)
	s = (2,2,4,4)
	ss = [(6,6,10,10)]
	assert not intersect_any(s, ss), "%s and %s should not intersect!" % (s,ss)
	s = (2,2,10,10)
	ss = [(4,4,8,8)]
	assert intersect_any(s, ss), "%s and %s should intersect!" % (s,ss)
	s = (4,4,8,8)
	ss = [(2,2,10,10)]
	assert intersect_any(s, ss), "%s and %s should intersect!" % (s,ss)
	s = (2,2,7,12)
	ss = [(5,5,10,10)]
	assert intersect_any(s, ss), "%s and %s should intersect!" % (s,ss)

#
#
# A square is (top, left, bottom, right)
#
def intersect(s1, s2):
	#
	# Find the top square.
	# It makes the checks easy and quick.
	#
	if s1[0] < s2[0]:
		t1, l1, b1, r1 = s1
		t2, l2, b2, r2 = s2
	else:
		t1, l1, b1, r1 = s2
		t2, l2, b2, r2 = s1
	# Check for side intersections
	if l2 <= l1 and l1 <= r2 and t1 <= t2 and b1 >= t2:
		return True
	if l2 <= r1 and r1 <= r2 and t1 <= t2 and b1 >= t2:
		return True
	# Check for bottom intersection
	if l1 <= l2 and r1 >= l2 and t2 <=b1:
		return True
	# Check for encompassing
	if t1 <= t2 and b1 >= b2 and l1 <= l2 and r1 >= r2:
		return True
	return False

def intersect_any(square, squares):
	for s in squares:
		Debug.Print("square: " + str(square))
		Debug.Print("s: " + str(s))
		if (intersect(square, s)):
			return True
	return False

class Debug:
	@classmethod
	def Print(cls, string):
		#print(string)
		pass

class GroundTruthError(Exception):
	def __init__(self, message):
		self.message = message

class Face:
	def __init__(self, expansion=0.25):
		self.left_eye = ()
		self.right_eye = ()
		self.left_mouth = ()
		self.right_mouth = ()
		self.center_mouth = ()
		self.nose = ()
		self.__square = None
		self.expansion = expansion

	def __repr__(self):
		return str(self.square())

	def square(self):
		if self.__square == None:
			# Well, nuts. The faces could be upside down.
			# We have to do this the hard way
			features = [self.left_eye, self.right_eye, self.left_mouth, self.nose, self.center_mouth, self.right_mouth];

			features = sorted(features, key=itemgetter(0))
			top = features[0][0]
			features = sorted(features, key=itemgetter(0), reverse=True)
			bottom = features[0][0]
			features = sorted(features, key=itemgetter(1))
			left = features[0][1]
			features = sorted(features, key=itemgetter(1), reverse=True)
			right = features[0][1]

			height_expansion = int(((bottom - top) * self.expansion) / 2.0)
			width_expansion = int(((right - left) * self.expansion) / 2.0)
			self.__square = (top - height_expansion,
				left - width_expansion,
				bottom + height_expansion,
				right + width_expansion)
		return self.__square

class GroundTruth:
	def __init__(self, data_dir, url=None, filename=None, expansion=None):
		self.data_dir = data_dir
		self.expansion = expansion
		self.face_count = 0
		self.faces = {}
		self.dimensions = {}
		if (url != None and filename != None) or\
		   (url == None and filename == None):
			raise GroundTruthError(
				"""Either url or filename (and not both) must be given."""
			)
		if url != None:
			self.data = self._download_ground_truth(url)
		else:
			self.data = self._read_ground_truth(filename)

		if self.data == "":
			return
		soup = BeautifulSoup(self.data)
		e = soup.find(['h1','pre'])
		dataset = ""
		dataitems = ""
		while e != None:
			if e.name == "h1":
				if re.search("Test Set A", e.get_text()):
					Debug.Print("Switching to A")
					dataset = "test"
				elif re.search("Test Set B", e.get_text()):
					Debug.Print("Switching to B")
					dataset = "test-low"
				elif re.search("Test Set C", e.get_text()):
					Debug.Print("Switching to C")
					dataset = "newtest"
				elif re.search("Rotated", e.get_text()):
					Debug.Print("Switching to Rotated")
					dataset = "rotated"
			elif e.name == "pre":
				for l in e.get_text().split("\n"):
					if l == "": continue
					parsed_line = self._parse_line(l)
					Debug.Print("line: " + str(l))
					Debug.Print("parsed line: " + str(parsed_line))
					self.face_count += 1
					if self.faces.has_key(dataset + "/" + parsed_line[0]):
						self.faces[dataset + "/" + parsed_line[0]].append(parsed_line[1])
					else:
						self.faces[dataset + "/" + parsed_line[0]] = [parsed_line[1]]

			e = e.findNext(['h1','pre'])

		for filename in self.faces.keys():
			image = skimage.img_as_float(skimage.io.imread(data_dir + "/" + filename))
			self.dimensions[filename] = image.shape

	def _parse_line(self, line):
		ls = line.split(" ")
		filename = ls[0]
		if self.expansion != None:
			face = Face(self.expansion)
		else:
			face = Face()
		face.left_eye = (int(float(ls[2])),int(float(ls[1])))
		face.right_eye = (int(float(ls[4])),int(float(ls[3])))
		face.nose = (int(float(ls[6])),int(float(ls[5])))
		face.left_mouth = (int(float(ls[8])),int(float(ls[7])))
		face.center_mouth = (int(float(ls[10])),int(float(ls[9])))
		face.right_mouth = (int(float(ls[12])),int(float(ls[11])))
		return (filename, face)

	def draw_face_squares(self, output_dir):
		for filename in self.faces.keys():
			image = skimage.img_as_float(skimage.io.imread(self.data_dir + filename))
			for face in self.faces[filename]:
				Debug.Print("filename: " + filename)
				Debug.Print("face: " + str(face))

				top, left, bottom, right = face.square()
				rr, cc = skimage.draw.line(top, left, top, right)
				image[rr,cc] = 1.0
				rr, cc = skimage.draw.line(top, right, bottom, right)
				image[rr,cc] = 1.0
				rr, cc = skimage.draw.line(bottom, right, bottom, left)
				image[rr,cc] = 1.0
				rr, cc = skimage.draw.line(bottom, left, top, left)
				image[rr,cc] = 1.0
			skimage.io.imsave(output_dir + "/" + os.path.split(filename)[1], image)

	def extract_random_patches(self, patch_shape, count=100):
		patches = []
		keys = self.faces.keys()
		for i in range(count):
			# Find a random image.
			filename = keys[random.randint(0,len(keys)-1)]
			faces = self.faces[filename]
			dimensions = self.dimensions[filename]
			Debug.Print("filename: " + str(filename))
			Debug.Print("faces: " + str([f.square() for f in faces]))
			Debug.Print("dimensions: " + str(dimensions))
			# Create a random patch
			while True:
				top = random.randint(0,dimensions[0]-1)
				left = random.randint(0,dimensions[1]-1)
				bottom = top + patch_shape[0]
				right = left + patch_shape[1]
				random_patch = (top, left, bottom, right)
				# Check that patch does not intersect with faces.
				if not intersect_any(random_patch, [f.square() for f in faces]):
					Debug.Print("random patch: " + str(random_patch))
					image = skimage.img_as_float(skimage.io.imread(self.data_dir + filename))
					patches.append((filename, i, image[top:bottom, left:right]))
					break
				# Repeat as necessary.
				pass
		"""
		for filename, counter, (top,left,bottom,right) in patches:
			image = skimage.img_as_float(skimage.io.imread("./data/" + filename))
			sample = image[top:bottom, left:right]
			skimage.io.imsave("output/" + counter + "-" + os.path.split(filename)[1], sample)
		"""
		return patches
	def extract_face_patches(self, count=120):
		patches = []
		total_counter = 0
		for filename in self.faces.keys():
			image = skimage.img_as_float(skimage.io.imread(self.data_dir + filename))
			per_file_count = 0
			for face in self.faces[filename]:

				Debug.Print("filename: " + filename)
				Debug.Print("face: " + str(face))
				Debug.Print("counter: " + str(per_file_count))

				top, left, bottom, right = face.square()
				patches.append((filename, per_file_count, image[top:bottom, left:right]))
				per_file_count+=1
				total_counter+=1
				if total_counter == count:
					return patches
		return patches

	def _download_ground_truth(self,url):
		req = urllib2.Request(url=url)
		try:
			f = urllib2.urlopen(req)
		except urllib2.URLError as e:
			return ""	
		return f.read()

	def _read_ground_truth(self,f):
		try:
			fh = open(f,'r')
		except IOError as e:
			return ""
		return fh.read()

def read_mosaic(mosaic_filename, (height,width)=(12,12), mosaic_width=12):
	patches = []
	image = skimage.img_as_float(skimage.io.imread(mosaic_filename))
	mosaic_height, mosaic_width = image.shape
	for y_spot in range(0,mosaic_height, height):
		for x_spot in range(0,mosaic_width, width):
			patches.append(image[y_spot:(y_spot+height), x_spot:(x_spot+width)])
	return patches

def generate_mosaic(patches, mosaic_filename,
                    (height, width)=(12,12),
										mosaic_width=12):
	counter = 0
	#
	# The len(patches) must be exactly a multiple of mosaic_width
	#
	mosaic = numpy.zeros(((height*((len(patches)/mosaic_width))),
		width*mosaic_width))
	for (filename, _, image) in patches:
		y_spot = counter/mosaic_width
		x_spot = counter%mosaic_width
		Debug.Print("y_spot: %d x_spot: %d filename: %s" % (y_spot, x_spot, filename))
		scaled = skimage.transform.resize(image, (height,width))
		mosaic[(y_spot*height):((y_spot*height)+height),\
		       (x_spot*width):((x_spot*width)+width)] =\
			scaled[0:height,0:width]
		counter += 1
	skimage.io.imsave(mosaic_filename, mosaic)

if __name__ == "__main__":
	unit_test_intersect()
