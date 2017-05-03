
CXX=g++
CXXFLAGS=-Wall


LIBS = -lrt
LDFLAGS = ${LIBS}


all: seq opencl

.PHONY: all seq opencl clean


seq: kmeans_seq

kmeans_seq: kmeans_seq.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS}


opencl: kmeans_opencl

kmeans_opencl: kmeans_opencl.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS} -lOpenCL


clean:
	rm -f kmeans kmeans_main.o kmeans_seq.o kmeans_opencl.o
