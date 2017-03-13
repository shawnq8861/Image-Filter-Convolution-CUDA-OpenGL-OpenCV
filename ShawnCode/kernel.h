#ifndef KERNEL_H
#define KERNEL_H

void modifyImage(unsigned char *imageMatrix, int rows, int cols, int radius, 
					unsigned char value);

void medianFilter(unsigned char *imageInMat, unsigned char *imageOutMat, 
					int rows, int cols, int kSize);

void boxFilter(unsigned char *imageInMat, unsigned char *imageOutMat, 
					int rows, int cols, int kSize);

#endif
