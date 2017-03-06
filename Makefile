###########################################
# 	Copyright (C) 2017
# 	Project : Matrix Convolution
# 	Author(s)  :
#       Description : Makefile 
###########################################

CC=nvcc
ARCH=sm_53
DEBUG_FLAGS=-g -lineinfo
CFLAGS= -arch $(ARCH) $(DEBUG_FLAGS)
LDFLAGS=-lopencv_core 
SRC=matconv.cu
TARGET=matconv
OBJ=$(SRC:.cu=.o)

.SUFFIXES: .cu .o

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) $(OBJ) -o $(TARGET)
.cu.o:
	$(CC) $(CFLAGS) $< -c -o $@
clean:
	rm -rf *.o $(TARGET)

