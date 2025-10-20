#include <CL/opencl.h>
#include <iostream>
#include <vector>
#include <string>

const char* kernelSource = R"(
__kernel void printNameDay(__global char* output) {
    int gid = get_global_id(0);

    const char days[7][12] = {"Sunday", "Monday", "Tuesday", "Wednesday",
                              "Thursday", "Friday", "Saturday"};

    int dayIdx = gid % 7;

    int offset = gid * 64;

    char temp[64];
    int pos = 0;

    temp[pos++] = 'T'; temp[pos++] = 'h'; temp[pos++] = 'r';
    temp[pos++] = 'e'; temp[pos++] = 'a'; temp[pos++] = 'd'; temp[pos++] = ' ';

    if (gid >= 10) {
        temp[pos++] = '0' + (gid / 10);
        temp[pos++] = '0' + (gid % 10);
    } else {
        temp[pos++] = '0' + gid;
    }

    temp[pos++] = ':'; temp[pos++] = ' ';
    temp[pos++] = 'T'; temp[pos++] = 'e'; temp[pos++] = 'j';
    temp[pos++] = 'a'; temp[pos++] = 'l'; temp[pos++] = ' ';
    temp[pos++] = '-'; temp[pos++] = ' ';

    for (int i = 0; days[dayIdx][i] != 0 && pos < 63; i++) {
        temp[pos++] = days[dayIdx][i];
    }
    temp[pos] = 0;

    for (int i = 0; i <= pos && i < 64; i++) {
        output[offset + i] = temp[i];
    }
}
)";

#define CL_CHECK(call) do { \
    cl_int err = call; \
    if (err != CL_SUCCESS) { \
        std::cerr << "OpenCL Error " << err << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {
    std::cout << "=== OpenCL Name and Day Program ===" << std::endl;

    try {
        cl_platform_id platform;
        cl_device_id device;

        CL_CHECK(clGetPlatformIDs(1, &platform, nullptr));

        cl_int result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (result != CL_SUCCESS) {
            std::cout << "GPU not found, using CPU..." << std::endl;
            CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr));
        } else {
            std::cout << "Using GPU device" << std::endl;
        }

        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &result);
        CL_CHECK(result);

        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &result);
        CL_CHECK(result);

        cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &result);
        CL_CHECK(result);

        result = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (result != CL_SUCCESS) {
            size_t logSize;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::vector<char> log(logSize);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
            std::cerr << "Build failed: " << log.data() << std::endl;
            CL_CHECK(result);
        }

        cl_kernel kernel = clCreateKernel(program, "printNameDay", &result);
        CL_CHECK(result);

        const size_t numThreads = 14;
        const size_t bufferSize = numThreads * 64;

        cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &result);
        CL_CHECK(result);

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &outputBuffer));

        std::cout << "Executing kernel with " << numThreads << " threads..." << std::endl;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &numThreads, nullptr, 0, nullptr, nullptr));
        CL_CHECK(clFinish(queue));

        std::vector<char> output(bufferSize, 0);
        CL_CHECK(clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, bufferSize, output.data(), 0, nullptr, nullptr));

        std::cout << "\n=== Output ===" << std::endl;
        for (size_t i = 0; i < numThreads; i++) {
            const char* line = &output[i * 64];
            if (line[0] != 0) {
                std::cout << line << std::endl;
            }
        }

        std::cout << "\n=== Analysis ===" << std::endl;
        std::cout << "- Each work item has unique global ID (0-13)" << std::endl;
        std::cout << "- Days cycle based on global ID % 7" << std::endl;
        std::cout << "- String formatting done in kernel" << std::endl;
        std::cout << "- Memory management between host/device" << std::endl;

        CL_CHECK(clReleaseMemObject(outputBuffer));
        CL_CHECK(clReleaseKernel(kernel));
        CL_CHECK(clReleaseProgram(program));
        CL_CHECK(clReleaseCommandQueue(queue));
        CL_CHECK(clReleaseContext(context));

        std::cout << "\nProgram completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}