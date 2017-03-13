#include "kernel.h"
#include <iostream>

//
// number of threads used to set block dimensions
//
#define NUM_THRDS 32

#define MAX_K_SIZE 9

using namespace std;

//
// define a square submatrix of size kSize
// using the thread index as the center value
// if the border values are outside the image matrix bounds,
// do not use those values in the calculation
// 	- sort the submatrix
//	- determine the median value of the sorted submatrix
//	- replace the original pixel value at index with the median value
//	
__global__ 
void medianFilterK(unsigned char *imageInMat, unsigned char *imageOutMat, 
				int rows, int cols, int kSize)
{
	unsigned char sortMatrix[MAX_K_SIZE * MAX_K_SIZE];
    // calculate the row and column that the thread works on
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;

	// use column major indexing to match OpenCV Mat class
	// the 1D thread index is the center pixel value of the kernel
	int pixel = col * rows + row;

	int radius = (kSize - 1)/2;

	// iterate around the submatrix and copy the values to be sorted	
	// to the sort matrix.  Do not copy values outside the image boundaries
	int count = 0;
	for (int c = col - radius; c < col + radius; ++c) {
		for (int r = row - radius; r < row + radius; ++r) {
			if (   (c < 0) 
				|| (r < 0)
				|| (c >= cols) 
				|| (r >= rows) ) {
				count += 0;
			}
			else {
				sortMatrix[count] = imageInMat[c * rows + r];
				count += 1;
			}
		} 
	}

	// run bubble sort only on the elements values added
	//
    // Conventional bubble sort
    //  1. loop through the array a number of times equal to the size of the 
	//	    array
    //  2. each iteration in each loop will compare two consecutive array 
	//	   elements
    //  3. if the first is greater than the second, swap the elements 
	//	   (sort ascending)
    //  4. alternatively, if the first is less than the second, swap the 
	//	   elements (sort descending)
    //

    // loop over the array...
    for (int i = 0; i < count; ++i) {
        // each iteration, ...
        for (int j = 0; j < count - 1; ++j) {
            // compare and swap...
            if (sortMatrix[j] > sortMatrix[j+1]) {
                int temp = sortMatrix[j];
                sortMatrix[j] = sortMatrix[j+1];
                sortMatrix[j+1] = temp;
            }
        }
    }
	int medIdx = (count / 2) + (count % 2);
	imageOutMat[pixel] = sortMatrix[medIdx];
}

//
// sum the elements defined by a square submatrix of size kSize
// using the thread index as the center value
// if the border values are outside the image matrix bounds,
// do not use those values in the calculation
//	
__global__ 
void boxFilterK(unsigned char *imageInMat, unsigned char *imageOutMat, 
				int rows, int cols, int kSize)
{
    // calculate the row and column that the thread works on
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;

	// use column major indexing to match OpenCV Mat class
	// the 1D thread index is the center pixel value of the kernel
	int pixel = col * rows + row;

	int radius = (kSize - 1)/2;

	// using the pixel value as the center anchor for the submatrix,
	// sum the kernel elements and divde the result by the number
	// of elements summed.  Do not include elements outside of the
	// image matrix.
	int sum = 0;
	int div = 0;
	for (int c = col - radius; c < col + radius; ++c) {
		for (int r = row - radius; r < row + radius; ++r) {
			if (   (c < 0) 
				|| (r < 0)
				|| (c >= cols) 
				|| (r >= rows) ) {
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
				 int rows, int cols, int kSize)
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
    medianFilterK<<<gridSize, blockSize>>>(d_imageInMat, d_imageOutMat, 
										rows, cols, kSize);

	// transfer GPU memory to CPU buffer
	cudaMemcpy(imageOutMat, d_imageOutMat, size, cudaMemcpyDeviceToHost);

	// free up the allocated memory
	cudaFree(d_imageInMat);
	cudaFree(d_imageOutMat);		
}
