#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <iostream>
#include <string.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

/**
@param string path      path to image to load
@param int mode         mode to open file as
@return Mat             returns the Mat of the loaded file
**/
Mat loadImage(string path, int mode) {
   Mat img = imread(path, mode);
   if (img.empty()) {
      cout << "error:  input image cannot be read..." << endl;
   }
   return img;
}

/**
@param string path      path to image to load
@return Mat             returns the Mat of the loaded file
**/
Mat loadImage(string path) {
   return loadImage(path, IMREAD_GRAYSCALE);
}

/**
@param string path      path to the file
@param Mat &img         image to save
@param int quality      quality to save image as
@return bool            true if file created, false otherwise
**/
bool saveImage(string path, Mat &img, int quality) {
   vector<int> compression_params;
   compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
   compression_params.push_back(quality);
   try {
      imwrite(path, img, compression_params);
   }
   catch (Exception& ex) {
      cout << "exception converting image to JPG format: "
         << ex.what() << endl;
      return false;
   }
   return true;
}

/**
@param string path      path to the file
@param Mat &img         image to save
@post-condition         uses 95 as image quality
@return bool            true if file created, false otherwise
**/
bool saveImage(string path, Mat &img) {
   return saveImage(path, img, 95);
}

/**
@param Mat &src         source
@param uchar *dst       destination
@precondition           output has been allocated with corret dimensions
TODO pad the rows to be multiple of four
**/
void mat2carry(Mat &src, unsigned char *dst) {
   int rows = src.rows, cols = src.cols;
   for (int col_local = 0; col_local < rows; ++col_local) {
      for (int row_local = 0; row_local < cols; ++row_local) {
         dst[cols * col_local + row_local] = src.at<uchar>(col_local, row_local);
      }
   }
}

/**
@param uchar *src       source
@param Mat &dst         destination
@precondition           output has been allocated with correct dimensions
TODO remove padding in conversion
**/
void carry2mat(unsigned char *src, Mat &dst) {
   int rows = dst.rows, cols = dst.cols;
   for (int col_local = 0; col_local < rows; ++col_local) {
      for (int row_local = 0; row_local < cols; ++row_local) {
         dst.at<uchar>(col_local, row_local) = src[cols * col_local + row_local];
      }
   }
}

/**
@param uchar *h_src     source on host
@param uchar *d_dst     destination on device
@param size_t size      size of memory to transfer
@return bool            true if allocation and transfer success, false otherwise
**/
bool allocateLoadMatGPU(unsigned char *h_src, unsigned char *d_dst, size_t size) {
   cudaError_t err1 = cudaMalloc((void **)&d_dst, size);
   if (err1 != cudaSuccess) {
      cout << "the error is " << cudaGetErrorString(err1) << endl;
      return false;
   }
   cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
   return true;
}

/**
@param uchar *d_src     source on device
@param uchar *h_dest    destination on host
@param size_t size      the size of memory to be transfered
**/
void loadMatCPU(unsigned char *d_src, unsigned char *h_dest, size_t size) {
   cudaMemcpy(h_dest, d_src, size, cudaMemcpyDeviceToHost);
}

#endif