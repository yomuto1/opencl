#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>

#include "sgemm.h"
#include "sgemm_common_def.h"

#define MAX_SOURCE_SIZE (0x100000)

int sgemm_alg(const float* __restrict p_a_f32, const float* __restrict p_b_f32, float* __restrict p_c_f32)
{
	for (int m = 0; m < SIZE_M; m++) {
		for (int n = 0; n < SIZE_N; n++) {
			float acc = 0.0f;
			for (int k = 0; k < SIZE_K; k++) {
				acc += p_a_f32[k * SIZE_M + m] * p_b_f32[n * SIZE_K + k];
			}
			p_c_f32[n * SIZE_M + m] = acc;
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
	cl_platform_id platformId = NULL;
	cl_device_id deviceID = NULL;
	cl_uint retNumDevices;
	cl_uint retNumPlatforms;
	cl_command_queue a_commandQueue[2];
	cl_mem a_aMemObj[2];
	cl_mem a_bMemObj[2];
	cl_mem a_cMemObj[2];
	cl_program program;
	cl_kernel kernel;
	const size_t local[2] = { TS_X, TS_Y };
	const size_t global[2] = { SIZE_M, SIZE_N };
	int m_s32 = SIZE_M;
	int n_s32 = SIZE_N;
	int k_s32 = SIZE_K;
	cl_event a_event[2];
	cl_int ret;
	int j_s32 = 0;
	int idx_0_s32 = 0;
	int idx_1_s32 = 0;

	fopen_s(&kernelFile, "sgemmKernel.cl", "r");

	if (NULL == kernelFile)
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
	ret = clGetPlatformIDs(1, &platformId, &retNumPlatforms);
	if (ret != CL_SUCCESS)
	{
		printf("clGetPlatformIDs failed! %d\n", ret);
		exit(-1);
	}

	ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceID, &retNumDevices);
	if (ret != CL_SUCCESS)
	{
		printf("clGetDeviceIDs failed! %d\n", ret);
		exit(-1);
	}

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
	if (ret != CL_SUCCESS)
	{
		printf("clCreateContext failed! %d\n", ret);
		exit(-1);
	}

	// Creating command queue
	a_commandQueue[0] = clCreateCommandQueueWithProperties(context, deviceID, 0, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("clCreateCommandQueueWithProperties failed! %d\n", ret);
		exit(-1);
	}
	a_commandQueue[1] = clCreateCommandQueueWithProperties(context, deviceID, 0, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("clCreateCommandQueueWithProperties failed! %d\n", ret);
		exit(-1);
	}

	// Memory buffers for each array
	a_aMemObj[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE_K * SIZE_M * sizeof(float), NULL, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("clCreateBuffer failed! %d\n", ret);
		exit(-1);
	}
	a_aMemObj[1] = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE_K * SIZE_M * sizeof(float), NULL, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("clCreateBuffer failed! %d\n", ret);
		exit(-1);
	}
	a_bMemObj[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE_K * SIZE_N * sizeof(float), NULL, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("clCreateBuffer failed! %d\n", ret);
		exit(-1);
	}
	a_bMemObj[1] = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE_K * SIZE_N * sizeof(float), NULL, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("clCreateBuffer failed! %d\n", ret);
		exit(-1);
	}
	a_cMemObj[0] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE_M * SIZE_N * sizeof(float), NULL, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("clCreateBuffer failed! %d\n", ret);
		exit(-1);
	}
	a_cMemObj[1] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE_M * SIZE_N * sizeof(float), NULL, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("clCreateBuffer failed! %d\n", ret);
		exit(-1);
	}

	// Create program from kernel source
	program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, (const size_t*)&kernelSize, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("clCreateProgramWithSource failed! %d\n", ret);
		exit(-1);
	}

	// Build program
	ret = clBuildProgram(program, 1, &deviceID, "-I ./", NULL, NULL);
	if (ret != CL_SUCCESS)
	{
		printf("clBuildProgram failed! %d\n", ret);
		exit(-1);
	}

	// Create kernel
	kernel = clCreateKernel(program, "myGEMM2", &ret);
	if (ret != CL_SUCCESS)
	{
		printf("clCreateKernel failed! %d\n", ret);
		exit(-1);
	}

	// Set arguments for kernel
	ret = clSetKernelArg(kernel, 0, sizeof(int), (void*)&m_s32);
	if (ret != CL_SUCCESS)
	{
		printf("clSetKernelArg failed! %d\n", ret);
		exit(-1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(int), (void*)&n_s32);
	if (ret != CL_SUCCESS)
	{
		printf("clSetKernelArg failed! %d\n", ret);
		exit(-1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(int), (void*)&k_s32);
	if (ret != CL_SUCCESS)
	{
		printf("clSetKernelArg failed! %d\n", ret);
		exit(-1);
	}

	// Copy lists to memory buffers
	ret = clEnqueueWriteBuffer(a_commandQueue[0], a_aMemObj[0], CL_FALSE, 0, SIZE_K * SIZE_M * sizeof(float), &p_a_f32[j_s32 * SIZE_K * SIZE_M], 0, NULL, NULL);;
	if (ret != CL_SUCCESS)
	{
		printf("clEnqueueWriteBuffer failed! %d\n", ret);
		exit(-1);
	}
	ret = clEnqueueWriteBuffer(a_commandQueue[0], a_bMemObj[0], CL_FALSE, 0, SIZE_K * SIZE_N * sizeof(float), &p_b_f32[j_s32 * SIZE_K * SIZE_N], 0, NULL, NULL);
	if (ret != CL_SUCCESS)
	{
		printf("clEnqueueWriteBuffer failed! %d\n", ret);
		exit(-1);
	}

	for (j_s32 = 0; j_s32 < (ITERATION - 1); ++j_s32)
	{
		idx_0_s32 = j_s32 % 2;
		idx_1_s32 = (j_s32 + 1) % 2;

		// Set arguments for kernel
		ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&a_aMemObj[idx_0_s32]);
		if (ret != CL_SUCCESS)
		{
			printf("clSetKernelArg failed! %d\n", ret);
			exit(-1);
		}
		ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&a_bMemObj[idx_0_s32]);
		if (ret != CL_SUCCESS)
		{
			printf("clSetKernelArg failed! %d\n", ret);
			exit(-1);
		}
		ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&a_cMemObj[idx_0_s32]);
		if (ret != CL_SUCCESS)
		{
			printf("clSetKernelArg failed! %d\n", ret);
			exit(-1);
		}

		// Execute the kernel
		ret = clEnqueueNDRangeKernel(a_commandQueue[idx_0_s32], kernel, 2, NULL, global, local, 0, NULL, &a_event[idx_0_s32]);
		if (ret != CL_SUCCESS)
		{
			printf("clEnqueueNDRangeKernel failed! %d\n", ret);
			exit(-1);
		}

		// Copy lists to memory buffers
		ret = clEnqueueWriteBuffer(a_commandQueue[idx_1_s32], a_aMemObj[idx_1_s32], CL_FALSE, 0, SIZE_K * SIZE_M * sizeof(float), &p_a_f32[(j_s32 + 1) * SIZE_K * SIZE_M], 0, NULL, &a_event[idx_1_s32]);
		if (ret != CL_SUCCESS)
		{
			printf("clEnqueueWriteBuffer failed! %d\n", ret);
			exit(-1);
		}
		ret = clEnqueueWriteBuffer(a_commandQueue[idx_1_s32], a_bMemObj[idx_1_s32], CL_FALSE, 0, SIZE_K * SIZE_N * sizeof(float), &p_b_f32[(j_s32 + 1) * SIZE_K * SIZE_N], 0, NULL, &a_event[idx_1_s32]);
		if (ret != CL_SUCCESS)
		{
			printf("clEnqueueWriteBuffer failed! %d\n", ret);
			exit(-1);
		}

		// Read from device back to host.
		ret = clEnqueueReadBuffer(a_commandQueue[idx_1_s32], a_cMemObj[idx_0_s32], CL_FALSE, 0, SIZE_M * SIZE_N * sizeof(float), &p_c_f32[j_s32 * SIZE_M * SIZE_N], 1, &a_event[idx_0_s32], NULL);
		if (ret != CL_SUCCESS)
		{
			printf("clEnqueueReadBuffer failed! %d\n", ret);
			exit(-1);
		}
	}

	idx_0_s32 = j_s32 % 2;
	idx_1_s32 = (j_s32 + 1) % 2;

	// Set arguments for kernel
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&a_aMemObj[idx_0_s32]);
	if (ret != CL_SUCCESS)
	{
		printf("clSetKernelArg failed! %d\n", ret);
		exit(-1);
	}
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&a_bMemObj[idx_0_s32]);
	if (ret != CL_SUCCESS)
	{
		printf("clSetKernelArg failed! %d\n", ret);
		exit(-1);
	}
	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&a_cMemObj[idx_0_s32]);
	if (ret != CL_SUCCESS)
	{
		printf("clSetKernelArg failed! %d\n", ret);
		exit(-1);
	}

	// Execute the kernel
	ret = clEnqueueNDRangeKernel(a_commandQueue[idx_0_s32], kernel, 2, NULL, global, local, 0, NULL, &a_event[idx_0_s32]);
	if (ret != CL_SUCCESS)
	{
		printf("clEnqueueNDRangeKernel failed! %d\n", ret);
		exit(-1);
	}

	// Read from device back to host.
	ret = clEnqueueReadBuffer(a_commandQueue[idx_1_s32], a_cMemObj[idx_0_s32], CL_TRUE, 0, SIZE_M * SIZE_N * sizeof(float), &p_c_f32[j_s32 * SIZE_M * SIZE_N], 1, &a_event[idx_0_s32], NULL);
	if (ret != CL_SUCCESS)
	{
		printf("clEnqueueReadBuffer failed! %d\n", ret);
		exit(-1);
	}

	// Clean up, release memory.
	ret = clFlush(a_commandQueue[0]);
	ret = clFlush(a_commandQueue[1]);
	ret = clFinish(a_commandQueue[0]);
	ret = clFinish(a_commandQueue[1]);
	ret = clReleaseCommandQueue(a_commandQueue[0]);
	ret = clReleaseCommandQueue(a_commandQueue[1]);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(a_aMemObj[0]);
	ret = clReleaseMemObject(a_aMemObj[1]);
	ret = clReleaseMemObject(a_bMemObj[0]);
	ret = clReleaseMemObject(a_bMemObj[1]);
	ret = clReleaseMemObject(a_cMemObj[0]);
	ret = clReleaseMemObject(a_cMemObj[1]);
	ret = clReleaseContext(context);

	return 0;
}
