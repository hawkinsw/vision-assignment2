clean:
	rm -f output/*
	rm -rf output2/
	rm -f *.pyc
	cd dataprep && make clean
