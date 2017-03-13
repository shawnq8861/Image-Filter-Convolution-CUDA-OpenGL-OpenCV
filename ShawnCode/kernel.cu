#include "kernel.h"
#include <iostream>

//
// number of threads used to set block dimensions
//
#define NUM_THRDS 32

using namespace std;

//
// apply the following filter to the image:
//		    1 1 1
//	(1/9) *	1 1 1
//		    1 1 1
//
// in this case of 3 X 3 filter kernel:
//			radius = 1
//	
__global__ 
void boxFilterK(unsigned char *imageInMat, unsigned char *imageOutMat, 
				int rows, int cols, int radius)
{
    // calculate the row and column that the thread works on
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;

	// use column major indexing to match OpenCV Mat class
	// the index is the center anchor for the kernel
	int pixel = col * rows + row;

	// using the index value as the center anchor for the kernel,
	// sum the products of the kernel elements and the input
	// image elements and divde the result by 1/9
	int sum = 0;
	int div = 0;
	for (int c = col - radius; c < col + radius; ++c) {
		for (int r = row - radius; r < row + radius; ++r) {
			if ((c < 0) || (r < 0)) {
				sum += 0;
			}
			else {
				sum += imageInMat[c * rows + r];
				div += 1;
			}
		} 
	}
	imageOutMat[pixel] = sum/div;
}

//
// set a square area inside image to a constant value
//
__global__ void modifyImageK(unsigned char *imageMatrix, int rows, 
							int cols, int radius, unsigned char value)
{
    // calculate the row and column that the thread works on
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;

	// use column major indexing to match OpenCV Mat class
	int index = col * rows + row;

	// define constraints on square region
    int rowCtr = rows/2 - 1;
	int colCtr = cols/2 - 1;
	int leftCol = colCtr - radius;
	int rightCol = colCtr + radius;
	int topRow = rowCtr - radius;
	int botRow = rowCtr + radius;

	if ( (row >= topRow)  
			&&
		 ( row <= botRow)
			&&
		 (col >= leftCol)
			&&
		 (col <= rightCol) ) {

		imageMatrix[index] = 0;

	}
	else {
		return;
	}
}

void modifyImage(unsigned char *imageMatrix, int rows, 
				 int cols, int radius, unsigned char value)
{
	// transfer CPU buffer to GPU memory and modify
	int size = rows * cols;
	unsigned char *d_imageMatrix;
	cudaError_t err1 = cudaMalloc((void **)&d_imageMatrix, size);
	if (err1 != cudaSuccess) {
        cout << "the error is " << cudaGetErrorString(err1) << endl;
    }
	cudaMemcpy(d_imageMatrix, imageMatrix, size, cudaMemcpyHostToDevice);

	// kernel call...

	// specify the 2D block dimensions in threads per block
    dim3 blockSize(NUM_THRDS, NUM_THRDS);

    // calculate the number of blocks per each grid side
    int bksX = (cols + blockSize.x - 1)/blockSize.x;
    int bksY = (rows + blockSize.y - 1)/blockSize.y;

    // specify the grid dimensions
    dim3 gridSize(bksX, bksY);

	// launch the modifyImageK() kernel function on the device (GPU)
    modifyImageK<<<gridSize, blockSize>>>(d_imageMatrix, rows, cols, 
											radius, value);

	// transfer GPU memory to CPU buffer
	cudaMemcpy(imageMatrix, d_imageMatrix, size, cudaMemcpyDeviceToHost);

	// free up the allocated memory
	cudaFree(d_imageMatrix);	
}

void filterImage(unsigned char *imageInMat, unsigned char *imageOutMat, 
				 int rows, int cols, int radius)
{
	// transfer CPU buffer to GPU memory and modify
	int size = rows * cols;
	unsigned char *d_imageInMat;
	cudaError_t err1 = cudaMalloc((void **)&d_imageInMat, size);
	if (err1 != cudaSuccess) {
        cout << "the error is " << cudaGetErrorString(err1) << endl;
    }
	unsigned char *d_imageOutMat;
	cudaError_t err2 = cudaMalloc((void **)&d_imageOutMat, size);
	if (err2 != cudaSuccess) {
        cout << "the error is " << cudaGetErrorString(err2) << endl;
    }
	cudaMemcpy(d_imageInMat, imageInMat, size, cudaMemcpyHostToDevice);

	// kernel call...

	// specify the 2D block dimensions in threads per block
    dim3 blockSize(NUM_THRDS, NUM_THRDS);

    // calculate the number of blocks per each grid side
    int bksX = (cols + blockSize.x - 1)/blockSize.x;
    int bksY = (rows + blockSize.y - 1)/blockSize.y;

    // specify the grid dimensions
    dim3 gridSize(bksX, bksY);

	// launch the modifyImageK() kernel function on the device (GPU)
    boxFilterK<<<gridSize, blockSize>>>(d_imageInMat, d_imageOutMat, 
										rows, cols, radius);

	// transfer GPU memory to CPU buffer
	cudaMemcpy(imageOutMat, d_imageOutMat, size, cudaMemcpyDeviceToHost);

	// free up the allocated memory
	cudaFree(d_imageInMat);
	cudaFree(d_imageOutMat);		
}
