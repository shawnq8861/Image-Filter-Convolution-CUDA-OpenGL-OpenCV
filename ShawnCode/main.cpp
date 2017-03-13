/*
	 nvcc readWriteJPGFile.cu -Wno-deprecated-gpu-targets -lopencv_core -lopencv_highgui -o readWriteJPGFile
*/

#include "kernel.h"
#include <iostream>
#include <string.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

int main(void)
{
    cout << "Hello CSS535 Final Project!" << endl;

    uchar *imageMatrix = NULL;
	uchar *rawImage = NULL;
	uchar *filteredImage = NULL;

    const string imageRoot =
    "/home/ubuntu/Documents/CSS535Projects/FinalProject";

    const string imageName = "/101_ObjectCategories/airplanes/image_0014.jpg";

    const string imagePath = imageRoot + imageName;

    Mat inImgMat = imread(imagePath, IMREAD_GRAYSCALE);
	// make a copy for later...
	Mat filterImgMat(inImgMat);
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

	// transfer Mat data to CPU image buffers
	imageMatrix = new uchar[size];
	rawImage = new uchar[size];
	filteredImage = new uchar[size];
	for (int col = 0; col < rows; ++col) {
		for (int row = 0; row < cols; ++row) {
			imageMatrix[cols * col + row] = inImgMat.at<uchar>(col, row);
			rawImage[cols * col + row] = inImgMat.at<uchar>(col, row);
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

	//
	// launch the modifyImageK() kernel function on the device (GPU)
	//
	uchar value = (uchar)0;
    modifyImage(imageMatrix, cols, rows, radius, value);

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

	//
	// launch the filterImageK() kernel function on the device (GPU)
	//
	int filterKernelSize = 3;
    filterImage(rawImage, filteredImage, cols, rows, filterKernelSize);
	
	// transfer CPU image buffer to Mat data
	for (int col = 0; col < rows; ++col) {
		for (int row = 0; row < cols; ++row) {
			filterImgMat.at<uchar>(col, row) = 
			filteredImage[cols * col + row];
		}
	}

	cout << "pixel value @ (vrtCtr, hrzCtr) = " << 
			(ushort)filterImgMat.at<uchar>(vrtCtr, hrzCtr) << endl;

	namedWindow("Kernel Filtered Image", WINDOW_AUTOSIZE);
    imshow("Kernel Filtered Image", inImgMat);

	// save kernel modified image to jpg file
	const string kFilterImgPath = imageRoot + "/kernelFilteredImage.jpg";
    try {
    	imwrite(kFilterImgPath, filterImgMat, compression_params);
    }
	catch (Exception& ex) {
    	cout << "exception converting image to JPG format: " 
			 << ex.what() << endl;
    	return 1;
    }
	
	delete[] imageMatrix;
	delete[] rawImage;
	delete[] filteredImage;

    waitKey();

	// close and destroy the open named windows
	destroyAllWindows();

    return 0;
}
