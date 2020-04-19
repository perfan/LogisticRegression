## Makefile for logisticRegression

SRC_DIR = ../../src/
RSLT_DIR = ./results/

all: clean run

clean-pyc:
	find . -name \*.pyc -type f -delete 

clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

clean-results:
	rm -rf ${RSLT_DIR}

clean: clean-pyc clean-results

run:
	mkdir ${RSLT_DIR};
	python3 logisticRegDNN.py 
	# python3 ${SRC_DIR}logisticRegDNN.py | tee -a ${RSLT_DIR}output.txt