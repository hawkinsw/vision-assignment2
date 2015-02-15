#!/usr/bin/python

from __future__ import print_function
from operator import itemgetter
import urllib2
from bs4 import BeautifulSoup
import skimage
import skimage.io
import skimage.draw
import skimage.transform
import re
import numpy


class Debug:
	@classmethod
	def Print(cls, string):
		#print(string)
		pass

class GroundTruthError(Exception):
	def __init__(self, message):
		self.message = message

class Face:
	def __init__(self, expansion=0.15):
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
	def __init__(self, url=None, filename=None):
		self.face_count = 0
		self.faces = {}
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
						self.faces[dataset + "/" + parsed_line[0]].append(parsed_line)
					else:
						self.faces[dataset + "/" + parsed_line[0]] = [parsed_line]

			e = e.findNext(['h1','pre'])

	def _parse_line(self, line):
		ls = line.split(" ")
		filename = ls[0]
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
			image = skimage.img_as_float(skimage.io.imread("./data/" + filename))
			for filename_face in self.faces[filename]:
				filename = filename_face[0]
				face = filename_face[1]
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
			skimage.io.imsave(output_dir + "/" + filename, image)

	def extract_face_patches(self):
		patches = []
		for filename in self.faces.keys():
			image = skimage.img_as_float(skimage.io.imread("./data/" + filename))
			counter = 0
			for filename_face in self.faces[filename]:
				filename = filename_face[0]
				face = filename_face[1]

				Debug.Print("filename: " + filename)
				Debug.Print("face: " + str(face))
				Debug.Print("counter: " + str(counter))

				top, left, bottom, right = face.square()
				patches.append((filename, counter, image[top:bottom, left:right]))
				counter+=1
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

def generate_face_mosaic(patches, mosaic_filename, height=12, width=12):
	counter = 0
	mosaic_width = 12
	mosaic = numpy.zeros(((height*(1+(len(patches)/mosaic_width))),
		width*mosaic_width))
	for (filename, _, image) in patches:
		y_spot = counter/mosaic_width
		x_spot = counter%mosaic_width
		print("y_spot: %d x_spot: %d" % (y_spot, x_spot))
		scaled = skimage.transform.resize(image, (height,width))
		mosaic[(y_spot*height):((y_spot*height)+height),\
		       (x_spot*width):((x_spot*width)+width)] =\
			scaled[0:height,0:width]
		counter += 1
	skimage.io.imsave(mosaic_filename, mosaic)

if __name__ == "__main__":
	d = GroundTruth(filename="./data.html")
	#d.draw_face_squares("./output")
	patches = d.extract_face_patches()
	generate_face_mosaic(patches, "./output/mosaic.gif")
