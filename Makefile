clean:
	rm -f output/*
	rm -f *.pyc
	cd dataprep && make clean
