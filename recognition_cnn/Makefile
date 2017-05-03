
CC=gcc
CFLAGS=-Wall -g #-O2

LIBS = -lm -lrt
LDFLAGS = ${LIBS}


all: seq opencl

.PHONY: all seq opencl clean


seq: recognition_seq

recognition_seq: recognition_seq.o main.o
	${CC} $^ -o $@ ${LDFLAGS}


opencl: recognition_opencl

recognition_opencl: recognition_opencl.o main.o 
	${CC} $^ -o $@ ${LDFLAGS} -lOpenCL


clean:
	rm -f recognition_seq.o recognition_opencl.o main.o recognition_seq recognition_opencl
