#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>

#include "sgemm.h"

#define MAX_SOURCE_SIZE (0x100000)

int sgemm_alg(const float* p_a_f32, const float* p_b_f32, float* p_c_f32)
{
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			float acc = 0.0f;
			for (int k = 0; k < K; k++) {
				acc += p_a_f32[k * M + m] * p_b_f32[n * K + k];
			}
			p_c_f32[n * M + m] = acc;
		}
	}

    return 0;
}

int sgemm_ocl(const float* p_a_f32, const float* p_b_f32, float* p_c_f32)
{
	// Load kernel from file vecAddKernel.cl
	FILE *kernelFile;
	char *kernelSource;
	size_t kernelSize;

	fopen_s(&kernelFile, "sgemmKernel.cl", "r");

	if (!kernelFile)
    {
		fprintf(stderr, "No file named sgemmKernel.cl was found\n");

		exit(-1);
	}

	kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
	if (NULL == kernelSource)
	{
		printf("malloc failed!\n");

		exit(-1);
	}
	kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
	fclose(kernelFile);

	// Getting platform and device information
	cl_platform_id platformId = NULL;
	cl_device_id deviceID = NULL;
	cl_uint retNumDevices;
	cl_uint retNumPlatforms;
	cl_int ret = clGetPlatformIDs(1, &platformId, &retNumPlatforms);
	ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceID, &retNumDevices);

#if 0
    char* value;
    size_t valueSize;
    cl_uint maxComputeUnits;
	cl_ulong localMemSize;
	cl_device_local_mem_type localMemType;
	size_t maxWorkGroupSize;

	printf("retNumPlatforms: %d, retNumDevices: %d\n", retNumPlatforms, retNumDevices);

	// print device name
	clGetDeviceInfo(deviceID, CL_DEVICE_NAME, 0, NULL, &valueSize);
	value = (char*) malloc(valueSize);
	clGetDeviceInfo(deviceID, CL_DEVICE_NAME, valueSize, value, NULL);
	printf("Device: %s\n", value);
	free(value);

	// print hardware device version
	clGetDeviceInfo(deviceID, CL_DEVICE_VERSION, 0, NULL, &valueSize);
	value = (char*) malloc(valueSize);
	clGetDeviceInfo(deviceID, CL_DEVICE_VERSION, valueSize, value, NULL);
	printf("Hardware version: %s\n", value);
	free(value);

	// print software driver version
	clGetDeviceInfo(deviceID, CL_DRIVER_VERSION, 0, NULL, &valueSize);
	value = (char*) malloc(valueSize);
	clGetDeviceInfo(deviceID, CL_DRIVER_VERSION, valueSize, value, NULL);
	printf("Software version: %s\n", value);
	free(value);

	// print c version supported by compiler for device
	clGetDeviceInfo(deviceID, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
	value = (char*) malloc(valueSize);
	clGetDeviceInfo(deviceID, CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
	printf("OpenCL C version: %s\n", value);
	free(value);

	// print parallel compute units
	clGetDeviceInfo(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS,
			sizeof(maxComputeUnits), &maxComputeUnits, NULL);
	printf("Parallel compute units: %d\n", maxComputeUnits);

	clGetDeviceInfo(deviceID, CL_DEVICE_LOCAL_MEM_SIZE,
		sizeof(localMemSize), &localMemSize, NULL);
	printf("Local mem size: %llu\n", localMemSize);

	clGetDeviceInfo(deviceID, CL_DEVICE_LOCAL_MEM_TYPE,
		sizeof(localMemType), &localMemType, NULL);
	printf("Local mem type: %u\n", localMemType);

	clGetDeviceInfo(deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE,
		sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
	printf("Max work group size: %llu\n", maxWorkGroupSize);
#endif

	// Creating context.
	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL,  &ret);


	// Creating command queue
	cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, deviceID, 0, &ret);

	// Memory buffers for each array
	cl_mem aMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, K * M * sizeof(float), NULL, &ret);
	cl_mem bMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY, K * N * sizeof(float), NULL, &ret);
	cl_mem cMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, M * N * sizeof(float), NULL, &ret);


	// Copy lists to memory buffers
	ret = clEnqueueWriteBuffer(commandQueue, aMemObj, CL_TRUE, 0, K * M * sizeof(float), p_a_f32, 0, NULL, NULL);;
	ret = clEnqueueWriteBuffer(commandQueue, bMemObj, CL_TRUE, 0, K * N * sizeof(float), p_b_f32, 0, NULL, NULL);

	// Create program from kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, (const size_t *)&kernelSize, &ret);	

	// Build program
	ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

	// Create kernel
	cl_kernel kernel = clCreateKernel(program, "myGEMM1", &ret);

	int m_s32 = M;
	int n_s32 = N;
	int k_s32 = K;

	// Set arguments for kernel
	ret = clSetKernelArg(kernel, 0, sizeof(int), (void*)&m_s32);
	ret = clSetKernelArg(kernel, 1, sizeof(int), (void*)&n_s32);
	ret = clSetKernelArg(kernel, 2, sizeof(int), (void*)&k_s32);
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&aMemObj);
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bMemObj);
	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&cMemObj);

	// Execute the kernel
	const int TS = 16;
	const size_t local[2] = { TS, TS };
	const size_t global[2] = { M, N };
	cl_event event;
	ret = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, global, local, 0, NULL, &event);
	ret = clWaitForEvents(1, &event);

	// Read from device back to host.
	ret = clEnqueueReadBuffer(commandQueue, cMemObj, CL_TRUE, 0, M * N * sizeof(float), p_c_f32, 0, NULL, NULL);

	// Clean up, release memory.
	ret = clFlush(commandQueue);
	ret = clFinish(commandQueue);
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(aMemObj);
	ret = clReleaseMemObject(bMemObj);
	ret = clReleaseMemObject(cMemObj);
	ret = clReleaseContext(context);

	return 0;
}
