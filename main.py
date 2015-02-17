#!/usr/bin/python

from dataprep import dataprep

if __name__=='__main__':
	d = dataprep.GroundTruth("./dataprep/data/", filename="./dataprep/data.html")
	#d.draw_face_squares("./output")
	patches = d.extract_face_patches(count=120)
	dataprep.generate_mosaic(patches, "./output/mosaic.gif")
