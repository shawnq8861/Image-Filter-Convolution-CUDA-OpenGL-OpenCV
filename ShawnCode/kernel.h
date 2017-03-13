#ifndef KERNEL_H
#define KERNEL_H

void modifyImage(unsigned char *imageMatrix, int rows, int cols, int radius, unsigned char value);

void filterImage(unsigned char *imageInMat, unsigned char *imageOutMat, int rows, int cols, int radius);

#endif

