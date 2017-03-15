#ifndef KERNEL_H
#define KERNEL_H

//
// number of threads used to set block dimensions
//
#define NUM_THRDS 32

//
// maxim height and width of filter kernels
//
#define MAX_K_SIZE 9

using namespace std;

//
// kernel launching function wrappers
//
void modifyImage(unsigned char *imageMatrix, int rows, int cols, int radius, 
					unsigned char value);

void medianFilter(unsigned char *imageInMat, unsigned char *imageOutMat, 
					int rows, int cols, int kSize);

void boxFilter(unsigned char *imageInMat, unsigned char *imageOutMat, 
					int rows, int cols, int kSize);


#endif

