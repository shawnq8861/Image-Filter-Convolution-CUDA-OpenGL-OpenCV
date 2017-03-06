//
// nvcc readWriteJPGFile.cu -Wno-deprecated-gpu-targets -lopencv_core 
// -lopencv_highgui -o readWriteJPGFile
//

#include <iostream>
#include <string.h>
#include <vector>
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
__global__ void modifyImageK(uchar *imageMatrix, int rows, 
							int cols, int radius, uchar value)
{
    // calculate the row and column that the thread works on
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	// column major indexing
	int col = index/rows;
	int row = index - (rows * col);
	// define constraints on square region
    int rowCtr = rows/2;
	int colCtr = cols/2;
	int leftCol = colCtr - radius;
	int rightCol = colCtr + radius;
	int topRow = rowCtr - radius;
	int botRow = rowCtr + radius;

	if ( (row > topRow)  
			&&
		 ( row < botRow)
			&&
		 (col > leftCol)
			&&
		 (col < rightCol) ) {

		imageMatrix[index] = 0;

	}
	else {
		return;
	}
}

int main(void)
{
    cout << "Hello CSS535 Final Project!" << endl;

    uchar *imageMatrix = NULL;

    const string imageRoot =
    "/home/ubuntu/Documents/CSS535Projects/FinalProject";

    const string imageName = "/101_ObjectCategories/airplanes/image_0014.jpg";

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

	// save original image to jpg file
	vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);
	const string origImgPath = imageRoot + "/originalImage.jpg";
    try {
    	imwrite(origImgPath, inImgMat, compression_params);
    }
	catch (Exception& ex) {
    	cout << "exception converting image to JPG format: " 
			 << ex.what() << endl;
    	return 1;
    }

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

	// save modified image to jpg file
	const string modImgPath = imageRoot + "/modifiedImage.jpg";
    try {
    	imwrite(modImgPath, inImgMat, compression_params);
    }
	catch (Exception& ex) {
    	cout << "exception converting image to JPG format: " 
			 << ex.what() << endl;
    	return 1;
    }

	// transfer CPU buffer to GPU memory and modify
	uchar *d_imageMatrix;
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
	uchar value = (uchar)0;
    modifyImageK<<<gridSize, blockSize>>>(d_imageMatrix, cols, rows, 
											radius, value);

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

	// save kernel modified image to jpg file
	const string kModImgPath = imageRoot + "/kernelModifiedImage.jpg";
    try {
    	imwrite(kModImgPath, inImgMat, compression_params);
    }
	catch (Exception& ex) {
    	cout << "exception converting image to JPG format: " 
			 << ex.what() << endl;
    	return 1;
    }
	
	delete[] imageMatrix;
	cudaFree(d_imageMatrix);

    waitKey();

    return 0;
}
