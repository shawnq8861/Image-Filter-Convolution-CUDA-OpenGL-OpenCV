#include "kernel.h"
#include <iostream>

//
// number of threads used to set block dimensions
//
#define NUM_THRDS 32

using namespace std;

//
// set a square area inside image to a constant value
//
__global__ void modifyImageK(unsigned char *imageMatrix, int rows, 
							int cols, int radius, unsigned char value)
{
    // calculate the row and column that the thread works on
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	// column major indexing
	int col = index/rows;
	int row = index - (rows * col);

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

    // 1D specification
    dim3 blockSize(NUM_THRDS);

    // calculate the number of blocks per each grid side
    // 1D specification
    int bksX = size/NUM_THRDS + 1;

    // specify the grid dimensions
    dim3 gridSize(bksX);

	// launch the modifyImageK() kernel function on the device (GPU)
	
    modifyImageK<<<gridSize, blockSize>>>(d_imageMatrix, rows, cols, 
											radius, value);

	// transfer GPU memory to CPU buffer
	cudaMemcpy(imageMatrix, d_imageMatrix, size, cudaMemcpyDeviceToHost);	
}

