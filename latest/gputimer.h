#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__

#pragma once


//Based on Cuda sample sdkTimer
class gpuTimer
{
private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
 
public:
    gpuTimer()
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
 
    ~gpuTimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
 
    void start()
    {
        cudaEventRecord(start_, 0);
    }
 
    void stop()
    {
        cudaEventRecord(stop_, 0);
    }
 
    float elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&elapsed, start_, stop_);
        return elapsed;
    }
};

#endif
