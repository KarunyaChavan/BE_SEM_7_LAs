#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

// Macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// CUDA kernel for array addition
__global__ void arrayAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Initialize array with random values
void initArray(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = (float)rand() / RAND_MAX * 100.0f;
    }
}

// Verify results
bool verify(const float* a, const float* b, const float* c, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(c[i] - (a[i] + b[i])) > 1e-5f) {
            std::cerr << "Error at index " << i << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 100000;
    const size_t size = N * sizeof(float);
    
    std::cout << "CUDA Array Addition (N=" << N << ")" << std::endl;
    
    // Host arrays
    std::vector<float> h_a(N), h_b(N), h_c(N);
    
    // Initialize random seed and arrays
    srand(time(nullptr));
    initArray(h_a.data(), N);
    initArray(h_b.data(), N);
    
    try {
        // Device arrays
        float *d_a, *d_b, *d_c;
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_a, size));
        CUDA_CHECK(cudaMalloc(&d_b, size));
        CUDA_CHECK(cudaMalloc(&d_c, size));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice));
        
        // Launch kernel
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        
        std::cout << "Launching kernel: " << numBlocks << " blocks, " 
                  << blockSize << " threads each" << std::endl;
        
        // Create events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        // Time the kernel execution
        CUDA_CHECK(cudaEventRecord(start));
        arrayAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Calculate time
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        std::cout << "Kernel time: " << ms << " ms" << std::endl;
        
        // Copy result back
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost));
        
        // Verify and display results
        if (verify(h_a.data(), h_b.data(), h_c.data(), N)) {
            std::cout << "SUCCESS: Verification passed!" << std::endl;
            std::cout << "\nFirst 5 results:" << std::endl;
            for (int i = 0; i < 5; i++) {
                std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
            }
        } else {
            std::cout << "FAILED: Verification failed!" << std::endl;
        }
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "Program completed successfully!" << std::endl;
    return EXIT_SUCCESS;
}
