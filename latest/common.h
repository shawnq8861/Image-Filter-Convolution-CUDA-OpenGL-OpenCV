#ifndef __COMMON_H__
#define __COMMON_H__

#pragma once
#include <opencv2/core/core.hpp>

// Gaussian kernel 5X5 
// Based on opencv gaussian kernel output for radius 5
const float gaussianKernel5x5[25] = 
{
    2.f/159.f,  4.f/159.f,  5.f/159.f,  4.f/159.f, 2.f/159.f,   
    4.f/159.f,  9.f/159.f, 12.f/159.f,  9.f/159.f, 4.f/159.f,   
    5.f/159.f, 12.f/159.f, 15.f/159.f, 12.f/159.f, 5.f/159.f,   
    4.f/159.f,  9.f/159.f, 12.f/159.f,  9.f/159.f, 4.f/159.f,   
    2.f/159.f,  4.f/159.f,  5.f/159.f,  4.f/159.f, 2.f/159.f,   
};

// Sobel kernel X gradient 
const float sobelGradientX[9] =
{
    -1.f, 0.f, 1.f,
    -2.f, 0.f, 2.f,
    -1.f, 0.f, 1.f,
};

// Sobel kernel Y gradient
const float sobelGradientY[9] =
{
    1.f,  2.f,  1.f,
    0.f,  0.f,  0.f,
   -1.f, -2.f, -1.f,
};

//Launcher
unsigned char* allocateBuffer(unsigned int size, unsigned char **dPtr);
void launchSobel_constantMemory(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,ssize_t offsetX,ssize_t offsetY);
void launchSobelNaive_withoutPadding(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,const float *d_X,const float *d_Y);
void launchSobelNaive_withPadding(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,const float *d_X,const float *d_Y);
void launchGaussian(unsigned char *dIn, unsigned char *dOut, cv::Size size,ssize_t offset);

//Kernel
__global__ void sobelGradientKernel(unsigned char *a, unsigned char *b, unsigned char *c);
__global__ void matrixConvGPU_constantMemory(unsigned char *dIn, int width, int height, int paddingX, int paddingY, ssize_t kOffset, int kernelW, int kernelH, unsigned char *dOut);
__global__ void matrixConvGPUNaive_withPadding(unsigned char *dIn, int width, int height, int paddingX, int paddingY, int kernelW, int kernelH, unsigned char *dOut, const float *kernel);
__global__ void matrixConvGPUNaive_withoutPadding(unsigned char *dIn, int width, int height, int kernelW, int kernelH, unsigned char *dOut, const float *kernel);

#endif
