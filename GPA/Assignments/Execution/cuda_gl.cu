// mandelbrot_zoom_cpu_gpu.cu
#include <windows.h>
#include <GL/gl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cmath>

#define WIDTH 1024
#define HEIGHT 1024
#define MAX_IT 1000

// ---------------- CPU ----------------
void mandelbrotCPU(unsigned char* img, int width, int height, double cx, double cy, double scale){
    for(int py=0; py<height; py++){
        for(int px=0; px<width; px++){
            double x0 = (px - width/2.0) * scale + cx;
            double y0 = (py - height/2.0) * scale + cy;
            double x=0,y=0;
            int iter=0;
            while(x*x+y*y<=4 && iter<MAX_IT){
                double xt = x*x - y*y + x0;
                y = 2*x*y + y0;
                x = xt;
                iter++;
            }
            int idx = 3*(py*width + px);
            unsigned char color = (unsigned char)(255*iter/MAX_IT);
            img[idx+0] = color;
            img[idx+1] = color;
            img[idx+2] = color;
        }
    }
}

// ---------------- GPU ----------------
__global__ void mandelbrotGPU(unsigned char* img, int width, int height, double cx, double cy, double scale){
    int px = blockIdx.x*blockDim.x + threadIdx.x;
    int py = blockIdx.y*blockDim.y + threadIdx.y;
    if(px>=width || py>=height) return;

    double x0 = (px - width/2.0) * scale + cx;
    double y0 = (py - height/2.0) * scale + cy;
    double x=0,y=0;
    int iter=0;
    while(x*x+y*y<=4 && iter<MAX_IT){
        double xt = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xt;
        iter++;
    }
    int idx = 3*(py*width + px);
    unsigned char color = (unsigned char)(255*iter/MAX_IT);
    img[idx+0] = color;
    img[idx+1] = color;
    img[idx+2] = color;
}

// ---------------- Display ----------------
void display(unsigned char* cpuImg, unsigned char* gpuImg){
    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2i(-1,-1);
    glDrawPixels(WIDTH/2, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, cpuImg);
    glRasterPos2i(0,-1);
    glDrawPixels(WIDTH/2, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, gpuImg);
    glFlush();
}

// ---------------- Win32 OpenGL ----------------
HWND createWindow(HINSTANCE hInstance, int width, int height){
    WNDCLASS wc={0};
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = DefWindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "MandelbrotZoomCPU_GPU";
    RegisterClass(&wc);
    return CreateWindowExA(0, wc.lpszClassName, "CPU vs GPU Mandelbrot Zoom",
        WS_OVERLAPPEDWINDOW|WS_VISIBLE, 100,100,width,height,nullptr,nullptr,hInstance,nullptr);
}

// ---------------- Main ----------------
int main(){
    HINSTANCE hInstance = GetModuleHandle(nullptr);
    HWND hwnd = createWindow(hInstance, WIDTH, HEIGHT);
    HDC hdc = GetDC(hwnd);

    PIXELFORMATDESCRIPTOR pfd={sizeof(PIXELFORMATDESCRIPTOR),1};
    pfd.dwFlags = PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    int pf = ChoosePixelFormat(hdc,&pfd);
    SetPixelFormat(hdc,pf,&pfd);
    HGLRC glrc = wglCreateContext(hdc);
    wglMakeCurrent(hdc,glrc);

    unsigned char* cpuImg = new unsigned char[3*WIDTH*HEIGHT/2];
    unsigned char* gpuImg = new unsigned char[3*WIDTH*HEIGHT/2];
    unsigned char* d_gpuImg;
    cudaMalloc(&d_gpuImg, 3*WIDTH*HEIGHT/2);

    double cx=0, cy=0, scale=4.0/WIDTH;

    MSG msg;
    int frame=0;
    bool running=true;
    while(running){
        while(PeekMessage(&msg,nullptr,0,0,PM_REMOVE)){
            if(msg.message==WM_QUIT) running=false;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        // Zoom in
        scale *= 0.99;
        cx += 0.001*frame;
        cy += 0.001*frame;

        // CPU timing
        auto startCPU = std::chrono::high_resolution_clock::now();
        mandelbrotCPU(cpuImg, WIDTH/2, HEIGHT, cx, cy, scale);
        auto endCPU = std::chrono::high_resolution_clock::now();
        double cpuTime = std::chrono::duration<double,std::milli>(endCPU-startCPU).count();

        // GPU timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        dim3 threads(16,16);
        dim3 blocks((WIDTH/2+15)/16,(HEIGHT+15)/16);
        mandelbrotGPU<<<blocks, threads>>>(d_gpuImg, WIDTH/2, HEIGHT, cx, cy, scale);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpuTime;
        cudaMemcpy(gpuImg, d_gpuImg, 3*WIDTH*HEIGHT/2, cudaMemcpyDeviceToHost);
        cudaEventElapsedTime(&gpuTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        display(cpuImg,gpuImg);
        SwapBuffers(hdc);

        std::cout << "Frame " << frame << ": CPU = " << cpuTime << " ms, GPU = " << gpuTime << " ms\n";
        frame++;
    }

    delete[] cpuImg;
    delete[] gpuImg;
    cudaFree(d_gpuImg);
    wglDeleteContext(glrc);
    ReleaseDC(hwnd,hdc);
    DestroyWindow(hwnd);
    cudaDeviceReset();
    return 0;
}
