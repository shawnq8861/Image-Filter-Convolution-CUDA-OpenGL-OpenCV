//
// g++ readWriteJPGFile.cpp -lopencv_core -lopencv_highgui -o readWriteJPGFile
//

#include <iostream>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

//
// number of threads used to set block dimensions
//
#define NUM_THRDS 32

using namespace std;
using namespace cv;

//
// set a square area inside image to a constant value
//
__global__ void modImageK(uchar *imageMatrix, int rows, int cols, uchar value)
{
    // calculate the row and column that the thread works on
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    int row = index/width;
    int col = index - (width * row);
    // verify that we are within the bounds of the matrices
    if ((row >= height) || (col >= width)) {
        return;
    }
    imageMatrix[cols * col + row] = value;
}

int main(void)
{
    cout << "Hello CSS535 Final Project!" << endl;

    uchar *imageMatrix = NULL;

    const string imageRoot =
    "/home/ubuntu/Documents/CSS535Projects/FinalProject/101_ObjectCategories";

    const string imageName = "/airplanes/image_0014.jpg";

    const string imagePath = imageRoot + imageName;

    Mat inImgMat = imread(imagePath, IMREAD_GRAYSCALE);
    if (inImgMat.empty()) {
        cout << "error:  input image cannot be read..." << endl;
    }

    int rows = inImgMat.rows;
    int cols = inImgMat.cols;
	uint bitDepth = inImgMat.depth();
	int size = rows * cols;

    cout << "rows = " << rows << endl;
    cout << "cols = " << cols << endl;
	if (bitDepth == CV_8U) {
		cout << "8 bit unsigned bit depth for the image" << endl;
	}

    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", inImgMat);

	int hrzCtr = cols/2;
	int vrtCtr = rows/2;

	// transfer Mat data to CPU image buffer
	imageMatrix = new uchar[size];
	for (int col = 0; col < rows; ++col) {
		for (int row = 0; row < cols; ++row) {
			imageMatrix[cols * col + row] = inImgMat.at<uchar>(col, row);
		}
	}

    // modify some values...
	
	cout << "pixel value @ (vrtCtr, hrzCtr) = " << 
			(ushort)inImgMat.at<uchar>(vrtCtr, hrzCtr) << endl;
	int radius = 30;
	for (int col = (vrtCtr - radius); col < (vrtCtr + radius); ++col) {
		for (int row = (hrzCtr - radius); row < (hrzCtr + radius); ++row) {
			imageMatrix[cols * col + row] = 255;
			//
			// two other ways of accessing the pixel value:
			//
			// *(imageMatrix + (cols * col + row)) = 255;
			// inImgMat.at<uchar>(col, row) = 255;
		}
	}

	// transfer CPU image buffer to Mat data
	for (int col = 0; col < rows; ++col) {
		for (int row = 0; row < cols; ++row) {
			inImgMat.at<uchar>(col, row) = imageMatrix[cols * col + row];
		}
	}

	cout << "pixel value @ (vrtCtr, hrzCtr) = " << 
			(ushort)inImgMat.at<uchar>(vrtCtr, hrzCtr) << endl;

	namedWindow("Modified Image", WINDOW_AUTOSIZE);
    imshow("Modified Image", inImgMat);


	// transfer CPU buffer to GPU memory and modify
	uchar *d_imageMatrix;
	cudaError_t err1 = cudaMalloc((void **)&d_imageMatrix, size);
	if (err1 != cudaSuccess) {
        cout << "the error is " << cudaGetErrorString(err1) << endl;
    }
	cudaMemcpy(d_imageMatrix, imageMatrix, size, cudaMemcpyHostToDevice);

	// kernel call

    // 1D specification
    dim3 blockSize(NUM_THRDS);

    // calculate the number of blocks per each grid side
    // 1D specification
    int bksX = (width * height)/NUM_THRDS + 1;

    // specify the grid dimensions
    dim3 gridSize(bksX);

	// launch the modifyImageK() kernel function on the device (GPU)
	uchar value = 0;
    matMatMultK<<<gridSize, blockSize>>>(d_imageMatrix, cols, rows, value);

	// transfer GPU memory to CPU buffer
	cudaMemcpy(imageMatrix, d_imageMatrix, size, cudaMemcpyDeviceToHost);

	// transfer CPU image buffer to Mat data
	for (int col = 0; col < rows; ++col) {
		for (int row = 0; row < cols; ++row) {
			inImgMat.at<uchar>(col, row) = imageMatrix[cols * col + row];
		}
	}

	cout << "pixel value @ (vrtCtr, hrzCtr) = " << 
			(ushort)inImgMat.at<uchar>(vrtCtr, hrzCtr) << endl;

	namedWindow("Kernel Modified Image", WINDOW_AUTOSIZE);
    imshow("Kernel Modified Image", inImgMat);
	
	delete[] imageMatrix;
	cudaFree(d_imageMatrix);

    waitKey();

    return 0;
}
