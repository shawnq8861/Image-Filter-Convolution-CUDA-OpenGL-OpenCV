#include <string>
#include <stdio.h>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include "gputimer.h"

__constant__ float constConvKernelMem[256];
// Create the cuda event timers 
gpuTimer timer;

using namespace std;

int main (int argc, char** argv)
{
    gpuTimer t1;
    double counter=0.0,tick=0.0;
    unsigned int frameCounter=0;
    float *d_X,*d_Y;

    /// Pass video file as input
    // For e.g. if camera device is at /dev/video1 - pass 1
    // You can pass video file as well instead of webcam stream
    cv::VideoCapture camera(1);
    
    cv::Mat frame;
    if(!camera.isOpened()) 
    {
        printf("Error .... campera not opened\n");;
        return -1;
    }
    
    // Open window for each kernel 
    cv::namedWindow("Source");
    cv::namedWindow("Grayscale");
    cv::namedWindow("Gaussian");
    cv::namedWindow("Sobel_constantMemory");
    cv::namedWindow("SobelNaive_withoutPadding");
    cv::namedWindow("SobelNaive_withPadding");


    cudaMemcpyToSymbol(constConvKernelMem, gaussianKernel5x5, sizeof(gaussianKernel5x5), 0);
    const ssize_t gaussianKernel5x5Offset = 0;

    cudaMemcpyToSymbol(constConvKernelMem, sobelGradientX, sizeof(sobelGradientX), sizeof(gaussianKernel5x5));
    cudaMemcpyToSymbol(constConvKernelMem, sobelGradientY, sizeof(sobelGradientY), sizeof(gaussianKernel5x5) + sizeof(sobelGradientX));
    
    // Calculate kernel offset in contant memory
    const ssize_t sobelKernelGradOffsetX = sizeof(gaussianKernel5x5)/sizeof(float);
    const ssize_t sobelKernelGradOffsetY = sizeof(sobelGradientX)/sizeof(float) + sobelKernelGradOffsetX;
 
    // Create matrix to hold original and processed image 
    camera >> frame;
    unsigned char *origData_d, *gaussianData_d, *sobelData_d_constMem, *sobelData_d_NaiveWithoutPadding, *sobelData_d_NaiveWithPadding;
    
    cudaMalloc((void **) &d_X, sizeof(sobelGradientX));
    cudaMalloc((void **) &d_Y, sizeof(sobelGradientY));
    cudaMemcpy(d_X, &sobelGradientX[0], sizeof(sobelGradientX), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, &sobelGradientY[0], sizeof(sobelGradientY), cudaMemcpyHostToDevice);

    cv::Mat origMat     (frame.size(), CV_8U, allocateBuffer(frame.size().width * frame.size().height, &origData_d));
    cv::Mat gaussianMat (frame.size(), CV_8U, allocateBuffer(frame.size().width * frame.size().height, &gaussianData_d));
    cv::Mat sobelMat    (frame.size(), CV_8U, allocateBuffer(frame.size().width * frame.size().height, &sobelData_d_constMem));
    cv::Mat sobelMatNaive_withoutPadding    (frame.size(), CV_8U, allocateBuffer(frame.size().width * frame.size().height, &sobelData_d_NaiveWithoutPadding));
    cv::Mat sobelMatNaive_withPadding    (frame.size(), CV_8U, allocateBuffer(frame.size().width * frame.size().height, &sobelData_d_NaiveWithPadding));

    // Create buffer to hold sobel gradients - XandY 
    unsigned char *sobelBufferX, *sobelBufferY;
    cudaMalloc(&sobelBufferX, frame.size().width * frame.size().height);
    cudaMalloc(&sobelBufferY, frame.size().width * frame.size().height);
    
    // Run loop to capture images from camera or loop over single image 
    while(1)
    {
        // Capture image frame 
        camera >> frame;
        
        // Convert frame to gray scale for further filter operation
	// Remove color channels, simplify convolution operation
        cv::cvtColor(frame, origMat, CV_BGR2GRAY);
        
        t1.start(); // timer for overall metrics
        launchGaussian(origData_d,gaussianData_d,frame.size(),gaussianKernel5x5Offset);
        launchSobel_constantMemory(gaussianData_d, sobelData_d_constMem, sobelBufferX, sobelBufferY, frame.size(), sobelKernelGradOffsetX, sobelKernelGradOffsetY);
        launchSobelNaive_withoutPadding(gaussianData_d, sobelData_d_NaiveWithoutPadding, sobelBufferX, sobelBufferY, frame.size(), d_X, d_Y);
        launchSobelNaive_withPadding(gaussianData_d, sobelData_d_NaiveWithPadding, sobelBufferX, sobelBufferY, frame.size(), d_X, d_Y);
        t1.stop();
        
        double tms = t1.elapsed(); 
        counter += tms*0.001;
        printf("Overall : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(frame.size().height*frame.size().width)/(tms*0.001),frame.size().height*frame.size().width,tms);
        
	// Update frame count 
        frameCounter +=1;
        if(counter - tick >= 1)
        {
            tick++;
            printf("Frames per second: %d\n",frameCounter);
            frameCounter = 0;
        }
        cv::imshow("Source", frame);
        cv::imshow("Grayscale", origMat);
        cv::imshow("Gaussian", gaussianMat);
        cv::imshow("Sobel_constantMemory", sobelMat);
        cv::imshow("SobelNaive_withoutPadding", sobelMatNaive_withoutPadding);
        cv::imshow("SobelNaive_withPadding", sobelMatNaive_withPadding);

        // Break loop
        if(cv::waitKey(1) == 27) break;
    }
    
    // Deallocate memory
    cudaFreeHost(origMat.data);
    cudaFreeHost(gaussianMat.data);
    cudaFreeHost(sobelMat.data);
    cudaFreeHost(sobelMatNaive_withoutPadding.data);
    cudaFreeHost(sobelMatNaive_withPadding.data);
    cudaFree(sobelBufferX);
    cudaFree(sobelBufferY);

    return 0;
}

void launchGaussian(unsigned char *dIn, unsigned char *dOut, cv::Size size,ssize_t offset)
{
    dim3 blocksPerGrid(size.width / 16, size.height / 16);
    dim3 threadsPerBlock(16, 16);
    
    // Perform the gaussian blur (first kernel in store @ 0)
    timer.start();
    {
         matrixConvGPU_constantMemory <<<blocksPerGrid,threadsPerBlock>>>(dIn,size.width, size.height, 0, 0, offset, 5, 5, dOut);
    }
    timer.stop();
    cudaThreadSynchronize();
    double tms = timer.elapsed(); 
    printf("Gaussian : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(tms*0.001),size.height*size.width,tms);
}


void launchSobel_constantMemory(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size,ssize_t offsetX,ssize_t offsetY)
{
    dim3 blocksPerGrid(size.width / 16, size.height / 16);
    dim3 threadsPerBlock(16, 16);
    
    // pythagoran kernel launch paramters
    dim3 blocksPerGridP(size.width * size.height / 256);
    dim3 threadsPerBlockP(256, 1);
     
    // Perform the sobel gradient convolutions (x&y padding is now 2 because there is a border of 2 around a 5x5 gaussian filtered image)
    timer.start();
    {
        matrixConvGPU_constantMemory<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 2, 2, offsetX, 3, 3, dGradX);
        matrixConvGPU_constantMemory<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 2, 2, offsetY, 3, 3, dGradY);
        sobelGradientKernel<<<blocksPerGridP,threadsPerBlockP>>>(dGradX, dGradY, dOut);
    }
    timer.stop();
    cudaThreadSynchronize();
    double tms = timer.elapsed(); 
    printf("Sobel (using constant memory) : Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(tms*0.001),size.height*size.width,tms);
}

void launchSobelNaive_withoutPadding(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size, const float *d_X,const float *d_Y)
{
    dim3 blocksPerGrid(size.width / 16, size.height / 16);
    dim3 threadsPerBlock(16, 16);
    
    // Dimension for Sobel gradient kernel 
    dim3 blocksPerGridP(size.width * size.height / 256);
    dim3 threadsPerBlockP(256, 1);
     
    timer.start();
    {
        matrixConvGPUNaive_withoutPadding<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 3, 3, dGradX,d_X);
        matrixConvGPUNaive_withoutPadding<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 3, 3, dGradY,d_Y);
        sobelGradientKernel<<<blocksPerGridP,threadsPerBlockP>>>(dGradX, dGradY, dOut);
    }
    timer.stop();
    cudaThreadSynchronize();
    double tms = timer.elapsed(); 
    printf("Sobel Naive (without padding): Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(tms*0.001),size.height*size.width,tms);
}

void launchSobelNaive_withPadding(unsigned char *dIn, unsigned char *dOut, unsigned char *dGradX, unsigned char *dGradY, cv::Size size, const float *d_X,const float *d_Y)
{
    dim3 blocksPerGrid(size.width / 16, size.height / 16);
    dim3 threadsPerBlock(16, 16);
    
    // Dimension for Sobel gradient kernel 
    dim3 blocksPerGridP(size.width * size.height / 256);
    dim3 threadsPerBlockP(256, 1);
     
    timer.start();
    {
        matrixConvGPUNaive_withPadding<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 2, 2, 3, 3, dGradX,d_X);
        matrixConvGPUNaive_withPadding<<<blocksPerGrid,threadsPerBlock>>>(dIn, size.width, size.height, 2, 2, 3, 3, dGradY,d_Y);
        sobelGradientKernel<<<blocksPerGridP,threadsPerBlockP>>>(dGradX, dGradY, dOut);
    }
    timer.stop();
    cudaThreadSynchronize();
    double tms = timer.elapsed(); 
    printf("Sobel Naive (with padding): Throughput in Megapixel per second : %.4f, Size : %d pixels, Elapsed time (in ms): %f\n",1.0e-6* (double)(size.height*size.width)/(tms*0.001),size.height*size.width,tms);
}

// Allocate buffer 
// Return ptr to shared mem
unsigned char* allocateBuffer(unsigned int size, unsigned char **dPtr)
{
    unsigned char *ptr = NULL;
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc(&ptr, size, cudaHostAllocMapped);
    cudaHostGetDevicePointer(dPtr, ptr, 0);
    return ptr;
}

// Used for Sobel edge detection
// Calculate gradient value from gradientX and gradientY  
// Calculate G = sqrt(Gx^2 * Gy^2)
__global__ void sobelGradientKernel(unsigned char *gX, unsigned char *gY, unsigned char *dOut)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    float x = float(gX[idx]);
    float y = float(gY[idx]);

    dOut[idx] = (unsigned char) sqrtf(x*x + y*y);
}

//naive without padding
__global__ void matrixConvGPUNaive_withoutPadding(unsigned char *dIn, int width, int height, int kernelW, int kernelH, unsigned char *dOut, const float *kernel) 
{
    // Pixel location 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float accum = 0.0;
    // Calculate radius along X and Y axis
    // We can also use one kernel variable instead - kernel radius
    int   kernelRadiusW = kernelW/2;
    int   kernelRadiusH = kernelH/2;

    // Determine pixels to operate 
    if(x >= kernelRadiusW && y >= kernelRadiusH &&
       x < (blockDim.x * gridDim.x) - kernelRadiusW &&
       y < (blockDim.y * gridDim.y)-kernelRadiusH)
    {
        for(int i = -kernelRadiusH; i <= kernelRadiusH; i++)  // Along Y axis
        {
            for(int j = -kernelRadiusW; j <= kernelRadiusW; j++) // Along X axis
            {
                // calculate weight 
                int jj = (j+kernelRadiusW);
                int ii = (i+kernelRadiusH);
                float w  = kernel[(ii * kernelW) + jj];
        
                accum += w * float(dIn[((y+i) * width) + (x+i)]);
            }
        }
    }
    
    dOut[(y * width) + x] = (unsigned char)accum;
}

//Naive with padding
__global__ void matrixConvGPUNaive_withPadding(unsigned char *dIn, int width, int height, int paddingX, int paddingY, int kernelW, int kernelH, unsigned char *dOut, const float *kernel)
{
    // Pixel location 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float accum = 0.0;
    // Calculate radius along X and Y axis
    // We can also use one kernel variable instead - kernel radius
    int   kernelRadiusW = kernelW/2;
    int   kernelRadiusH = kernelH/2;

    // Determine pixels to operate 
    if(x >= (kernelRadiusW + paddingX) && y >= (kernelRadiusH + paddingY) &&
       x < ((blockDim.x * gridDim.x) - kernelRadiusW - paddingX) &&
       y < ((blockDim.y * gridDim.y) - kernelRadiusH - paddingY))
    {
        for(int i = -kernelRadiusH; i <= kernelRadiusH; i++)  // Along Y axis
        {
            for(int j = -kernelRadiusW; j <= kernelRadiusW; j++) // Along X axis
            {
                // calculate weight 
                int jj = (j+kernelRadiusW);
                int ii = (i+kernelRadiusH);
                float w  = kernel[(ii * kernelW) + jj];
        
                accum += w * float(dIn[((y+i) * width) + (x+i)]);
            }
        }
    }
    
    dOut[(y * width) + x] = (unsigned char)accum;
}

//Constant memory
__global__ void matrixConvGPU_constantMemory(unsigned char *dIn, int width, int height, int paddingX, int paddingY, ssize_t kernelOffset, int kernelW, int kernelH, unsigned char *dOut)
{
    // Calculate our pixel's location
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Calculate radius along X and Y axis
    // We can also use one kernel variable instead - kernel radius
    float accum = 0.0;
    int   kernelRadiusW = kernelW/2;
    int   kernelRadiusH = kernelH/2;

    // Determine pixels to operate 
    if(x >= (kernelRadiusW + paddingX) && y >= (kernelRadiusH + paddingY) &&
       x < ((blockDim.x * gridDim.x) - kernelRadiusW - paddingX) &&
       y < ((blockDim.y * gridDim.y) - kernelRadiusH - paddingY))
    {
        for(int i = -kernelRadiusH; i <= kernelRadiusH; i++) // Along Y axis
        {
            for(int j = -kernelRadiusW; j <= kernelRadiusW; j++) //Along X axis
            {
                // Sample the weight for this location
                int jj = (j+kernelRadiusW);
                int ii = (i+kernelRadiusH);
                float w  = constConvKernelMem[(ii * kernelW) + jj + kernelOffset]; //kernel from constant memory
                 
                accum += w * float(dIn[((y+i) * width) + (x+j)]);
            }
        }
    }
    
    dOut[(y * width) + x] = (unsigned char) accum;
}
