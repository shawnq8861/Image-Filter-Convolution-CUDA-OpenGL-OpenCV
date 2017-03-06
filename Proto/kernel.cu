#include "kernel.h"
#define TX 32
#define TY 32

//TODO update kernel and launcher

__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__global__
void distanceKernel(unsigned char *d_out, int w, int h, int2 pos) {
   const int c = blockIdx.x*blockDim.x + threadIdx.x;
   const int r = blockIdx.y*blockDim.y + threadIdx.y;
   if ((c >= w) || (r >= h)) return; // Check if within image bounds
   const int i = c + r*w; // 1D indexing
   d_out[i]++;
}

void kernelLauncher(unsigned char *d_out, int w, int h, int2 pos) {
   const dim3 blockSize(TX, TY);
   const dim3 gridSize = dim3((w + TX - 1) / TX, (h + TY - 1) / TY);
   distanceKernel << <gridSize, blockSize >> >(d_out, w, h, pos);
}