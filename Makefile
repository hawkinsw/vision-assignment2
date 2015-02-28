clean:
	rm -f test_output/*
	rm -f output/*
	rm -rf output2/
	rm -f *.pyc
	cd dataprep && make clean
