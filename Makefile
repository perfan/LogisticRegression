## Makefile for logisticRegression

SRC_DIR = ./src/

clean-pyc:
	find . -name \*.pyc -type f -delete 

clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

clean: clean-pyc

run:
	python3 ${SRC_DIR}logisticRegDNN.py 