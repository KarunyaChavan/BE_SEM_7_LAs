// matrix_mult_cuda.cu
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" void cuda_matmul(double *A,double *B,double *C,int N);

__global__ void matmul_kernel(const double *A,const double *B,double *C,int N){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<N && col<N){
        double sum=0;
        for(int k=0;k<N;k++) sum+=A[row*N+k]*B[k*N+col];
        C[row*N+col]=sum;
    }
}

void cuda_matmul(double *A,double *B,double *C,int N){
    double *dA,*dB,*dC;
    size_t bytes=N*N*sizeof(double);
    cudaMalloc(&dA,bytes); cudaMalloc(&dB,bytes); cudaMalloc(&dC,bytes);
    cudaMemcpy(dA,A,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,bytes,cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks((N+15)/16,(N+15)/16);
    matmul_kernel<<<blocks,threads>>>(dA,dB,dC,N);
    cudaDeviceSynchronize();

    cudaMemcpy(C,dC,bytes,cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
