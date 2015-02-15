#!/usr/bin/python

from __future__ import print_function
import urllib2
from bs4 import BeautifulSoup
import skimage
import skimage.io
import skimage.draw
import re


class Debug:
	@classmethod
	def Print(cls, string):
		#print(string)
		pass

class GroundTruthError(Exception):
	def __init__(self, message):
		self.message = message

class Face:
	def __init__(self):
		self.left_eye = ()
		self.right_eye = ()
		self.left_mouth = ()
		self.right_mouth = ()
		self.center_mouth = ()
		self.nose = ()
		self.__square = None

	def square(self):
		if self.__square == None:
			# Find smallest y
			if self.left_eye[0] < self.right_eye[0]:
				top = self.left_eye[0]
			else:
				top = self.right_eye[0]
			# Find smallest x
			if self.left_eye[1] < self.left_mouth[1]:
				left = self.left_eye[1]
			else:
				left = self.left_mouth[1]
			# Find biggest x
			if self.left_mouth[1] > self.right_mouth[1]:
				right = self.left_mouth[1]
			else:
				right = self.right_mouth[1]
			# Find biggest y
			if self.left_mouth[0] > self.right_mouth[0]:
				bottom = self.left_mouth[0]
			else:
				bottom = self.right_mouth[0]
			self.__square = (top, left, bottom, right)
		return self.__square

class GroundTruth:
	def __init__(self, url=None, filename=None):
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
					if self.faces.has_key(dataset + "/" + parsed_line[0]):
						self.faces[dataset + "/" + parsed_line[0]].append(parsed_line)
					else:
						self.faces[dataset + "/" + parsed_line[0]] = [parsed_line]

			e = e.findNext(['h1','pre'])
		print("Done")

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
				print("filename: " + filename)
				print("face: " + str(face))
				rr, cc = skimage.draw.line(*(face.square()))
				image[rr,cc] = 1.0
				rr, cc = skimage.draw.line(*(face.left_eye + face.right_eye))
				image[rr,cc] = 1.0
				rr, cc = skimage.draw.line(*(face.left_eye + face.left_mouth))
				image[rr,cc] = 1.0
				rr, cc = skimage.draw.line(*(face.right_eye + face.right_mouth))
				image[rr,cc] = 1.0
				rr, cc = skimage.draw.line(*(face.left_mouth + face.right_mouth))
				image[rr,cc] = 1.0
			skimage.io.imsave(output_dir + "/" + filename, image)

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

if __name__ == "__main__":
	d = GroundTruth(filename="./data.html")
	d.draw_face_squares("./output")
