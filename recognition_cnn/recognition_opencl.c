#include <stdio.h>
#include <string.h>
#include "recognition.h"
#include <CL/cl.h>

#include <time.h>

#define RUN_TIME_KERNEL_BUILD   (0) /* 0 ==> bin-run or pre build mode, 1 ==> run-time compile mode */
#define PRE_BUILD_MODE          (2 & (0 == RUN_TIME_KERNEL_BUILD)) /* 0 ==> bin run mode, 1 ==> pre build mode */

#define FILE_NAME_KERNEL_CODE_LARGE   "recognition_large.cl"
#define FILE_NAME_KERNEL_BIN_LARGE    "recognition_large.bin"
#define FILE_NAME_KERNEL_CODE_MEDIUM  "recognition_medium.cl"
#define FILE_NAME_KERNEL_BIN_MEDIUM   "recognition_medium.bin"
#define FILE_NAME_KERNEL_CODE_SMALL   "recognition_small.cl"
#define FILE_NAME_KERNEL_BIN_SMALL    "recognition_small.bin"

#define RUN_WITH_CL_CODE   (0)
#define PRE_BUILD_COMPILE  (1)
#define RUN_WITH_BINARY    (2)

#if(1 == RUN_TIME_KERNEL_BUILD)
#define RUN_MODE  (RUN_WITH_CL_CODE)
#else
#if (1 == PRE_BUILD_MODE)
#define RUN_MODE  (PRE_BUILD_COMPILE)
#else
#define RUN_MODE  (RUN_WITH_BINARY)
#endif
#endif

#define DEBUGGING_INFO_PRINT (0)
#define CHECK_BUILD_ERROR_LOG (0)
#define PROFILING_ENABLE (0)
#define PINGPONG_ENABLE (0)

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

#if (RUN_WITH_BINARY != RUN_MODE)
static char *get_source_code(const char *file_name, size_t *len);
#endif
#if (1 == DEBUGGING_INFO_PRINT)
static void print_device_name(cl_device_id dev);
#endif
#if (1 == PROFILING_ENABLE)
static int timespec_subtract(struct timespec*, struct timespec*, struct timespec*);
#endif
  
#if (RUN_WITH_BINARY != RUN_MODE)
static char *get_source_code(const char *file_name, size_t *len)
{
  char *source_code;
  size_t length;
  FILE *file = fopen(file_name, "r");
  if (file == NULL) {
    printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
    exit(EXIT_FAILURE);
  }

  fseek(file, 0, SEEK_END);
  length = (size_t)ftell(file);
  rewind(file);

  source_code = (char *)malloc(length + 1);
  fread(source_code, length, 1, file);
  source_code[length] = '\0';

  fclose(file);

  *len = length;
  return source_code;
}
#endif

#if (1 == DEBUGGING_INFO_PRINT)
static void print_device_name(cl_device_id dev) {
  cl_int err;
  size_t name_size;

  clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &name_size);

  char *dev_name = (char *)malloc(name_size + 1);
  err = clGetDeviceInfo(dev, CL_DEVICE_NAME,
                        name_size, dev_name, NULL);
  CHECK_ERROR(err);

  printf("Device: %s\n", dev_name);

  free(dev_name);
}
#endif

#if (1 == PROFILING_ENABLE)
static int timespec_subtract(struct timespec* result, struct timespec *x, struct timespec *y) {
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_nsec < y->tv_nsec) {
    int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
    y->tv_nsec -= 1000000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_nsec - y->tv_nsec > 1000000000) {
    int nsec = (x->tv_nsec - y->tv_nsec) / 1000000000;
    y->tv_nsec += 1000000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_nsec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_nsec = x->tv_nsec - y->tv_nsec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}
#endif

#define SZ_MAX_CU       (32)
#define SZ_MAX_PE       (64)
#define SZ_MAX_PES_ALL  (SZ_MAX_CU * SZ_MAX_PE)
#define SZ_LOCAL (SZ_MAX_PE)
#define SZ_GLOBAL (SZ_MAX_PES_ALL)

#define SZ_NO_TASK_IN_WG (4)
	
void recognition(float * images, float * network, int depth, int size, int * labels, float * confidences)
{
	/* start of local variable declaration */

	/* CL platform information */
	cl_platform_id *platform;
	cl_device_type dev_type = CL_DEVICE_TYPE_GPU;
	cl_device_id *devs;
	cl_context context;
	cl_command_queue *cmd_queues;
	cl_program *program;

	cl_uint num_platforms;
	cl_uint num_devs = 0;
	cl_int err;
#if (1 == PROFILING_ENABLE)
	cl_ulong queued_time, submit_time, start_time, end_time;
#endif

	size_t sz_local_recognition   = SZ_LOCAL;
	size_t sz_global_recognition  = SZ_GLOBAL;

	int i;
	float **weights, **biases;
	
#if (1 == PROFILING_ENABLE)
	struct timespec start, end, spent;
	clock_gettime(CLOCK_MONOTONIC, &start);
#endif

	weights = (float **)malloc(sizeof(float *) * (depth + 1));
	biases = (float **)malloc(sizeof(float *) * (depth + 1));

	/* Set pointers for weights and biases */
	/* 1. Input layer */
	weights[0] = network;
	biases[0] = weights[0] + size * IMG_SIZE;
	/* 2. Hidden layers */
	for(i = 1; i < depth; i++)
	{
		weights[i] = network + (size * IMG_SIZE + size) + (size * size + size) * (i-1);
		biases[i] = weights[i] + size * size;
	}
	/* 3. Output layer */
	weights[depth] = weights[depth - 1] + size * size + size;
	biases[depth] = weights[depth] + DIGIT_COUNT * size;

	char *dtype = getenv("CL_DEV_TYPE");

	/* end of local variable declaration */
	
	/* get the device type to use from the environmental variable */
	if(dtype)
	{
		if(strcasecmp(dtype, "cpu") == 0)
		{
			dev_type = CL_DEVICE_TYPE_CPU;
		}
		else if(strcasecmp(dtype, "gpu") == 0)
		{
			dev_type = CL_DEVICE_TYPE_GPU;
		}
		else
		{
			CHECK_ERROR(-1);
		}
	}	

	/* get platform IDs */
	err = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_ERROR(err);
	if(num_platforms == 0)
	{
		CHECK_ERROR(-1);
	}
	platform = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
	err = clGetPlatformIDs(num_platforms, platform, NULL);
	CHECK_ERROR(err);
#if (1 == DEBUGGING_INFO_PRINT)
	printf("Number of platforms: %u\n", num_platforms);
#endif
#if (1 == PROFILING_ENABLE)
	clock_gettime(CLOCK_MONOTONIC, &end);
	timespec_subtract(&spent, &end, &start);
	printf("clGetPlatformIDs time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif

	/* get device IDs */
	for(i = 0; i < num_platforms; i++)
	{
		err = clGetDeviceIDs(platform[i], dev_type, 0, NULL, &num_devs);
		if(err != CL_DEVICE_NOT_FOUND)
		{
			CHECK_ERROR(err);
		}
		if(num_devs >= 1)
		{
			devs = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devs);
			
			err = clGetDeviceIDs(platform[i], dev_type, num_devs, devs, NULL);
			CHECK_ERROR(err);
			
			break;
		}
	}
	
	if((devs == NULL) || (num_devs < 1))
	{
		CHECK_ERROR(-1);
	}
#if (1 == DEBUGGING_INFO_PRINT)
	for(i = 0; i < num_devs; i++)
	{
		printf("dev[%d] : ", i);
		print_device_name(devs[i]);
	}
#endif
#if (1 == PROFILING_ENABLE)
	clock_gettime(CLOCK_MONOTONIC, &end);
	timespec_subtract(&spent, &end, &start);
	printf("clGetDeviceIDs time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif

	/* create context */
	context = clCreateContext(NULL, num_devs, devs, NULL, NULL, &err);
	CHECK_ERROR(err);

#if (1 == DEBUGGING_INFO_PRINT)
	printf("clCreateContext done\n");
#endif
#if (1 == PROFILING_ENABLE)
	clock_gettime(CLOCK_MONOTONIC, &end);
	timespec_subtract(&spent, &end, &start);
	printf("clCreateContext time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif

	/* create in-order cmd queue */
	cmd_queues = (cl_command_queue *)malloc(sizeof(cl_command_queue) * num_devs);
	for(i = 0; i < num_devs; i++)
	{
#if (1 == PROFILING_ENABLE)
		cmd_queues[i] = clCreateCommandQueue(context, devs[i], CL_QUEUE_PROFILING_ENABLE, &err);
#else
		cmd_queues[i] = clCreateCommandQueue(context, devs[i], 0, &err);
#endif
		CHECK_ERROR(err);
		
#if (1 == DEBUGGING_INFO_PRINT)
		printf("clCreateCommandQueue %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
		clock_gettime(CLOCK_MONOTONIC, &end);
		timespec_subtract(&spent, &end, &start);
		printf("clCreateCommandQueue %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
	}

	program = (cl_program *)malloc(sizeof(cl_program) * num_devs);

	if(size == 64) /* small network */
	{
#if (RUN_WITH_CL_CODE == RUN_MODE)
		size_t sz_kernel_src;
		char *code_kernel_src = get_source_code(FILE_NAME_KERNEL_CODE_SMALL, &sz_kernel_src);
#elif (PRE_BUILD_COMPILE == RUN_MODE)
		size_t sz_kernel_src;
		char *code_kernel_src = get_source_code(FILE_NAME_KERNEL_CODE_SMALL, &sz_kernel_src);
#else /* RUN_WITH_BINARY == RUN_MODE) */
#endif

		/* kernel object */
		cl_kernel *kernel_inp_lyr  = NULL;
		cl_kernel *kernel_hdd1_lyr  = NULL;
		cl_kernel *kernel_out_lyr  = NULL;
		cl_kernel *kernel_red_lyr  = NULL;

		/* memory object */
		cl_mem *p_inp_lyr_data_fp32 = NULL;
		cl_mem *p_inp_hdd0_lyr_wgt_conv_fp32 = NULL;
		cl_mem *p_inp_hdd0_lyr_wgt_bias_fp32 = NULL;

		cl_mem *p_ino_hdd0_lyr_data_fp32 = NULL;
		cl_mem *p_inp_hdd1_lyr_wgt_conv_fp32 = NULL;
		cl_mem *p_inp_hdd1_lyr_wgt_bias_fp32 = NULL;

		cl_mem *p_ino_hdd1_lyr_data_fp32 = NULL;
		cl_mem *p_inp_hdd2_lyr_wgt_conv_fp32 = NULL;
		cl_mem *p_inp_hdd2_lyr_wgt_bias_fp32 = NULL;

		cl_mem *p_ino_hdd2_lyr_data_fp32 = NULL;

		cl_mem *p_out_label_s32 = NULL;
		cl_mem *p_out_conf_lv_fp32 = NULL;	

		/* event */
		cl_event *ev_write = NULL;
		cl_event *ev_kernel_i = NULL, *ev_kernel_1 = NULL, *ev_kernel_2 = NULL, *ev_kernel_r = NULL;
		cl_event *ev_read = NULL;

		/* create buffer object */
		p_inp_lyr_data_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd0_lyr_wgt_conv_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd0_lyr_wgt_bias_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_ino_hdd0_lyr_data_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd1_lyr_wgt_conv_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd1_lyr_wgt_bias_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_ino_hdd1_lyr_data_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd2_lyr_wgt_conv_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd2_lyr_wgt_bias_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_ino_hdd2_lyr_data_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_out_label_s32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_out_conf_lv_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		for(i = 0; i < num_devs; i++)
		{
			p_inp_lyr_data_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * IMG_SIZE * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd0_lyr_wgt_conv_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * IMG_SIZE * size, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd0_lyr_wgt_bias_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size, NULL, &err);
			CHECK_ERROR(err);
			
			p_ino_hdd0_lyr_data_fp32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * size * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd1_lyr_wgt_conv_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size * size, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd1_lyr_wgt_bias_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size, NULL, &err);
			CHECK_ERROR(err);
			
			p_ino_hdd1_lyr_data_fp32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * size * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd2_lyr_wgt_conv_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size * DIGIT_COUNT, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd2_lyr_wgt_bias_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * DIGIT_COUNT, NULL, &err);
			CHECK_ERROR(err);
			
			p_ino_hdd2_lyr_data_fp32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * DIGIT_COUNT * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			
			p_out_label_s32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);		
			p_out_conf_lv_fp32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clCreateBuffer %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clCreateBuffer %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}

		ev_write = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_kernel_i = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_kernel_1 = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_kernel_2 = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_kernel_r = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_read = (cl_event *)malloc(sizeof(cl_event) * num_devs);		
		
		/* write buffer */
		for(i = 0; i < num_devs; i++)
		{
#if (1 == PROFILING_ENABLE)
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_lyr_data_fp32[i], CL_FALSE, 0, sizeof(cl_float) * IMG_SIZE * IMG_COUNT / num_devs, &images[i * IMG_SIZE * IMG_COUNT / num_devs], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_lyr_data_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd0_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * IMG_SIZE * size, weights[0], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd0_lyr_wgt_conv_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd0_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[0], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd0_lyr_wgt_bias_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd1_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * size, weights[1], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd1_lyr_wgt_conv_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd1_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[1], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd1_lyr_wgt_bias_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd2_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * DIGIT_COUNT, weights[2], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd2_lyr_wgt_conv_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd2_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * DIGIT_COUNT, biases[2], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd2_lyr_wgt_bias_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
#else
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_lyr_data_fp32[i], CL_FALSE, 0, sizeof(cl_float) * IMG_SIZE * IMG_COUNT / num_devs, &images[i * IMG_SIZE * IMG_COUNT / num_devs], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd0_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * IMG_SIZE * size, weights[0], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd0_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[0], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd1_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * size, weights[1], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd1_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[1], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd2_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * DIGIT_COUNT, weights[2], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd2_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * DIGIT_COUNT, biases[2], 0, NULL, NULL);
			CHECK_ERROR(err);
#endif

#if (1 == DEBUGGING_INFO_PRINT)
			printf("clEnqueueWriteBuffer %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clEnqueueWriteBuffer %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}
		
#if (RUN_WITH_CL_CODE == RUN_MODE)
		/* create program object */
		for(i = 0; i < num_devs; i++)
		{
			program[i] = clCreateProgramWithSource(context, 1, (const char **)&code_kernel_src, &sz_kernel_src, &err);
			CHECK_ERROR(err);
		}
		
		free(code_kernel_src);
		
		/* build kernel source code */
		for(i = 0; i < num_devs; i++)
		{
#if (1 == CHECK_BUILD_ERROR_LOG)
			size_t log_size = 0;
			char *log = NULL;
#endif
			err = clBuildProgram(program[i], 1, &devs[i], "", NULL, NULL);
#if (1 == CHECK_BUILD_ERROR_LOG)
			clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			log = (char *)malloc(log_size + 1);
			clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
			log[log_size] = '\0';
			printf(":kernel Compile log:\n%s\n", log);
			free(log);
			log = NULL;
#endif
			CHECK_ERROR(err);
		}
#elif (PRE_BUILD_COMPILE == RUN_MODE)
		/* create program object */
		for(i = 0; i < num_devs; i++)
		{
			program[i] = clCreateProgramWithSource(context, 1, (const char **)&code_kernel_src, &sz_kernel_src, &err);
			CHECK_ERROR(err);
		}
		
		/* build kernel source code */
		for(i = 0; i < num_devs; i++)
		{
#if (1 == CHECK_BUILD_ERROR_LOG)
			size_t log_size = 0;
			char *log = NULL;
#endif
			err = clBuildProgram(program[i], 1, &devs[0], "", NULL, NULL);
#if (1 == CHECK_BUILD_ERROR_LOG)
			clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			log = (char *)malloc(log_size + 1);
			clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
			log[log_size] = '\0';
			printf(":kernel Compile log:\n%s\n", log);
			free(log);
			log = NULL;
#endif
			CHECK_ERROR(err);
		}
    
		size_t nbread;
		size_t *np = (size_t *)malloc(sizeof(size_t) * num_devs); /* Create size array */
		err = clGetProgramInfo(program[0], CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * num_devs, np, &nbread); /* Load in np the size of my binary */
		CHECK_ERROR(err);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("num_devs: %d nbread: %d np: %d %d %d %d\n", num_devs, nbread, np[0], np[1], np[2], np[3]);
#endif
		char** bn = (char **)malloc(sizeof(char *) * num_devs); /* Create the binary array */
		for(i = 0; i < num_devs; i++)
		{
			bn[i] = (char *)malloc(sizeof(char) * np[i]); /* I know... it's bad... but if i use new char[np[i]], i have a segfault... */ 
		}
		err = clGetProgramInfo(program[0], CL_PROGRAM_BINARIES, sizeof(unsigned char *)*num_devs, bn, &nbread); //Load the binary itself  
		CHECK_ERROR(err);
		FILE *fp = fopen(FILE_NAME_KERNEL_BIN_SMALL, "wb");
		i=0;
#if (1 == DEBUGGING_INFO_PRINT)
		printf("%s\n", bn[i]);
#endif
		fwrite(bn[i], sizeof(char), np[i], fp); // Save the binary, but my file stay empty  
		fclose(fp);  

		free(np);
		free(bn);

		printf("\n pre-build done !\n change PRE_BUILD_MODE mode as 0\n");
		exit(-1);
#else /* RUN_WITH_BINARY == RUN_MODE) */
		FILE *fp = fopen(FILE_NAME_KERNEL_BIN_SMALL, "rb");
		size_t binarySize[4];
		cl_int binaryStatus[4];
		fseek(fp, 0, SEEK_END);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("fseek done\n");
#endif
		binarySize[0] = ftell(fp);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("binarySize done: %d\n", binarySize[0]);
#endif
		rewind(fp);
		unsigned char *programBinary = (unsigned char *)malloc(binarySize[0]);
		fread(programBinary, 1, binarySize[0], fp);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("programBinary done: %d %d %d %d\n", programBinary[0], programBinary[1], programBinary[2], programBinary[3]);
#endif
		binarySize[0] = binarySize[0];
		binarySize[1] = binarySize[0];
		binarySize[2] = binarySize[0];
		binarySize[3] = binarySize[0];
#if (1 == DEBUGGING_INFO_PRINT)
		printf("binarySize: %d %d %d %d\n", binarySize[0], binarySize[1], binarySize[2], binarySize[3]);
#endif
		for(i = 0; i < num_devs; i++)
		{
			program[i] = clCreateProgramWithBinary(context, 1, &devs[i], &binarySize[i], (const unsigned char**)&programBinary, &binaryStatus[i], &err);
			CHECK_ERROR(err);			
		}
#if (1 == DEBUGGING_INFO_PRINT)
		printf("clCreateProgramWithBinary done: %u %u %u %u\n", binaryStatus[0], binaryStatus[1], binaryStatus[2], binaryStatus[3]);
#endif
		CHECK_ERROR(err);
		fclose(fp);

		/* build kernel source code */
		for(i = 0; i < num_devs; i++)
		{
			err = clBuildProgram(program[i], 1, &devs[i], "", NULL, NULL);
			CHECK_ERROR(err);
		}
#if (1 == DEBUGGING_INFO_PRINT)
		printf("clBuildProgram done\n");
#endif
#if (1 == CHECK_BUILD_ERROR_LOG)
		{
			size_t log_size = 0;
			char *log = NULL;
			for(i = 0; i < num_devs; i++)
			{
				clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
				log = (char *)malloc(log_size + 1);
				clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
				log[log_size] = '\0';
				printf(":kernel Compile log:\n%s\n", log);
				free(log);
				log = NULL;
			}
		}
#endif
		CHECK_ERROR(err);
#endif 

#if (1 == DEBUGGING_INFO_PRINT)
		printf("clBuildProgram done\n");
#if (1 == PROFILING_ENABLE)
		clock_gettime(CLOCK_MONOTONIC, &end);
		timespec_subtract(&spent, &end, &start);
		printf("clBuildProgram time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
#endif

		/* create kernel object */
		kernel_inp_lyr = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devs);
		kernel_hdd1_lyr = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devs);
		kernel_out_lyr = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devs);
		kernel_red_lyr = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devs);
		for(i = 0; i < num_devs; i++)
		{
			kernel_inp_lyr[i] = clCreateKernel(program[i], "kernel_sml_inp_lyr", &err);
			CHECK_ERROR(err);			
			kernel_hdd1_lyr[i] = clCreateKernel(program[i], "kernel_sml_hdn_lyr", &err);
			CHECK_ERROR(err);
			kernel_out_lyr[i] = clCreateKernel(program[i], "kernel_sml_out_lyr", &err);
			CHECK_ERROR(err);
			kernel_red_lyr[i] = clCreateKernel(program[i], "kernel_reduction_lyr", &err);
			CHECK_ERROR(err);
			
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clCreateKernel %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clCreateKernel %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}

		/* kernel argument setting */
		for(i = 0; i < num_devs; i++)
		{
			err = clSetKernelArg(kernel_inp_lyr[i], 0, sizeof(cl_mem), &p_inp_lyr_data_fp32[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(kernel_inp_lyr[i], 1, sizeof(cl_mem), &p_inp_hdd0_lyr_wgt_conv_fp32[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(kernel_inp_lyr[i], 2, sizeof(cl_mem), &p_inp_hdd0_lyr_wgt_bias_fp32[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(kernel_inp_lyr[i], 3, sizeof(cl_mem), &p_ino_hdd0_lyr_data_fp32[i]);
			CHECK_ERROR(err);

			err = clSetKernelArg(kernel_hdd1_lyr[i], 0, sizeof(cl_mem), &p_ino_hdd0_lyr_data_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd1_lyr[i], 1, sizeof(cl_mem), &p_inp_hdd1_lyr_wgt_conv_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd1_lyr[i], 2, sizeof(cl_mem), &p_inp_hdd1_lyr_wgt_bias_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd1_lyr[i], 3, sizeof(cl_mem), &p_ino_hdd1_lyr_data_fp32[i]);
			CHECK_ERROR(err);

			err = clSetKernelArg(kernel_out_lyr[i], 0, sizeof(cl_mem), &p_ino_hdd1_lyr_data_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_out_lyr[i], 1, sizeof(cl_mem), &p_inp_hdd2_lyr_wgt_conv_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_out_lyr[i], 2, sizeof(cl_mem), &p_inp_hdd2_lyr_wgt_bias_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_out_lyr[i], 3, sizeof(cl_mem), &p_ino_hdd2_lyr_data_fp32[i]);
			CHECK_ERROR(err);

			err = clSetKernelArg(kernel_red_lyr[i], 0, sizeof(cl_mem), &p_ino_hdd2_lyr_data_fp32[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(kernel_red_lyr[i], 1, sizeof(cl_mem), &p_out_label_s32[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(kernel_red_lyr[i], 2, sizeof(cl_mem), &p_out_conf_lv_fp32[i]);
			CHECK_ERROR(err);
			
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clSetKernelArg %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clSetKernelArg %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}
		
		/* run kernel */
		for(i = 0; i < num_devs; i++)
		{
			err = clEnqueueNDRangeKernel(cmd_queues[i], kernel_inp_lyr[i], 1, NULL, &sz_global_recognition, &sz_local_recognition, 0, NULL, &ev_kernel_i[i]);
			CHECK_ERROR(err);
			err = clEnqueueNDRangeKernel(cmd_queues[i], kernel_hdd1_lyr[i], 1, NULL, &sz_global_recognition, &sz_local_recognition, 0, NULL, &ev_kernel_1[i]);
			CHECK_ERROR(err);
			err = clEnqueueNDRangeKernel(cmd_queues[i], kernel_out_lyr[i], 1, NULL, &sz_global_recognition, &sz_local_recognition, 0, NULL, &ev_kernel_2[i]);
			CHECK_ERROR(err);

			err = clEnqueueNDRangeKernel(cmd_queues[i], kernel_red_lyr[i], 1, NULL, &sz_global_recognition, &sz_local_recognition, 0, NULL, &ev_kernel_r[i]);
			CHECK_ERROR(err);

#if (1 == DEBUGGING_INFO_PRINT)
			printf("kernel %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("kernel %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif			
		}

		/* read buffer */
		for(i = 0; i < num_devs; i++)
		{
#if (1 == PROFILING_ENABLE)
			err = clEnqueueReadBuffer(cmd_queues[i], p_out_label_s32[i], CL_TRUE, 0, sizeof(cl_int) * IMG_COUNT / num_devs, &labels[i * IMG_COUNT / num_devs], 1, &ev_kernel_r[i], &ev_read[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueReadBuffer p_out_label_s32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueReadBuffer(cmd_queues[i], p_out_conf_lv_fp32[i], CL_TRUE, 0, sizeof(cl_float) * IMG_COUNT / num_devs, &confidences[i * IMG_COUNT / num_devs], 1, &ev_kernel_r[i], &ev_read[i]);
			CHECK_ERROR(err);			
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueReadBuffer p_out_conf_lv_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
#else
			err = clEnqueueReadBuffer(cmd_queues[i], p_out_label_s32[i], CL_TRUE, 0, sizeof(cl_int) * IMG_COUNT / num_devs, &labels[i * IMG_COUNT / num_devs], 1, &ev_kernel_r[i], NULL);
			CHECK_ERROR(err);
			err = clEnqueueReadBuffer(cmd_queues[i], p_out_conf_lv_fp32[i], CL_TRUE, 0, sizeof(cl_float) * IMG_COUNT / num_devs, &confidences[i * IMG_COUNT / num_devs], 1, &ev_kernel_r[i], NULL);
			CHECK_ERROR(err);			
#endif

#if (1 == DEBUGGING_INFO_PRINT)
			printf("clEnqueueReadBuffer %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clEnqueueReadBuffer %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}

#if (1 == PROFILING_ENABLE)
		/* kernel profile results print */
		for(i = 0; i < num_devs; i++)
		{
			err = clGetEventProfilingInfo(ev_kernel_i[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_i[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_i[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_i[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueNDRangeKernel kernel_inp_lyr %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clGetEventProfilingInfo(ev_kernel_1[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_1[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_1[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_1[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueNDRangeKernel kernel_hdd1_lyr %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clGetEventProfilingInfo(ev_kernel_2[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_2[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_2[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_2[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueNDRangeKernel kernel_out_lyr %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);			
			err = clGetEventProfilingInfo(ev_kernel_r[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_r[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_r[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_r[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueNDRangeKernel kernel_red_lyr %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);			
			
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clGetEventProfilingInfo %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clGetEventProfilingInfo %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}
#endif		
		
		/* release stage */
		for(i = 0; i < num_devs; i++)
		{
			clReleaseMemObject(p_inp_lyr_data_fp32[i]);
			clReleaseMemObject(p_inp_hdd0_lyr_wgt_conv_fp32[i]);
			clReleaseMemObject(p_inp_hdd0_lyr_wgt_bias_fp32[i]);
			clReleaseMemObject(p_ino_hdd0_lyr_data_fp32[i]);
			clReleaseMemObject(p_inp_hdd1_lyr_wgt_conv_fp32[i]);
			clReleaseMemObject(p_inp_hdd1_lyr_wgt_bias_fp32[i]);
			clReleaseMemObject(p_ino_hdd1_lyr_data_fp32[i]);
			clReleaseMemObject(p_inp_hdd2_lyr_wgt_conv_fp32[i]);
			clReleaseMemObject(p_inp_hdd2_lyr_wgt_bias_fp32[i]);
			clReleaseMemObject(p_ino_hdd2_lyr_data_fp32[i]);
			clReleaseMemObject(p_out_label_s32[i]);
			clReleaseMemObject(p_out_conf_lv_fp32[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseMemObject %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseMemObject %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseKernel(kernel_inp_lyr[i]);
			clReleaseKernel(kernel_hdd1_lyr[i]);
			clReleaseKernel(kernel_out_lyr[i]);
			clReleaseKernel(kernel_red_lyr[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseKernel %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseKernel %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
#if (1 == PROFILING_ENABLE)
			clReleaseEvent(ev_write[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_write[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_write[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
#endif
#if (1 == PROFILING_ENABLE)
			clReleaseEvent(ev_kernel_i[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_kernel_0[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_kernel_0[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseEvent(ev_kernel_1[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_kernel_1[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_kernel_1[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseEvent(ev_kernel_2[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_kernel_2[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_kernel_2[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseEvent(ev_kernel_r[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_kernel_r[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_kernel_r[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseEvent(ev_read[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_read[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_read[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif		
#endif
			clReleaseProgram(program[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseProgram %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseProgram %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}

		free(kernel_inp_lyr);
		free(kernel_hdd1_lyr);
		free(kernel_out_lyr);
		free(kernel_red_lyr);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("free kernel done\n");
#endif
#if (1 == PROFILING_ENABLE)
		clock_gettime(CLOCK_MONOTONIC, &end);
		timespec_subtract(&spent, &end, &start);
		printf("free kernel time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		free(p_inp_lyr_data_fp32);
		free(p_inp_hdd0_lyr_wgt_conv_fp32);
		free(p_inp_hdd0_lyr_wgt_bias_fp32);
		free(p_ino_hdd0_lyr_data_fp32);
		free(p_inp_hdd1_lyr_wgt_conv_fp32);
		free(p_inp_hdd1_lyr_wgt_bias_fp32);
		free(p_ino_hdd1_lyr_data_fp32);
		free(p_inp_hdd2_lyr_wgt_conv_fp32);
		free(p_inp_hdd2_lyr_wgt_bias_fp32);
		free(p_ino_hdd2_lyr_data_fp32);
		free(p_out_label_s32);
		free(p_out_conf_lv_fp32);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("free mem done\n");
#endif
#if (1 == PROFILING_ENABLE)
		clock_gettime(CLOCK_MONOTONIC, &end);
		timespec_subtract(&spent, &end, &start);
		printf("free mem time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
	}
	else if(size == 512) /* medium network */
	{
#if (RUN_WITH_CL_CODE == RUN_MODE)
		size_t sz_kernel_src;
		char *code_kernel_src = get_source_code(FILE_NAME_KERNEL_CODE_MEDIUM, &sz_kernel_src);
#elif (PRE_BUILD_COMPILE == RUN_MODE)
		size_t sz_kernel_src;
		char *code_kernel_src = get_source_code(FILE_NAME_KERNEL_CODE_MEDIUM, &sz_kernel_src);
#else /* RUN_WITH_BINARY == RUN_MODE) */
#endif

		/* kernel object */
		cl_kernel *kernel_inp_lyr  = NULL;
		cl_kernel *kernel_hdd1_lyr  = NULL;
		cl_kernel *kernel_hdd2_lyr  = NULL;
		cl_kernel *kernel_out_lyr  = NULL;
		cl_kernel *kernel_red_lyr  = NULL;

		/* memory object */
		cl_mem *p_inp_lyr_data_fp32 = NULL;
		cl_mem *p_inp_hdd0_lyr_wgt_conv_fp32 = NULL;
		cl_mem *p_inp_hdd0_lyr_wgt_bias_fp32 = NULL;

		cl_mem *p_ino_hdd0_lyr_data_fp32 = NULL;
		cl_mem *p_inp_hdd1_lyr_wgt_conv_fp32 = NULL;
		cl_mem *p_inp_hdd1_lyr_wgt_bias_fp32 = NULL;

		cl_mem *p_ino_hdd1_lyr_data_fp32 = NULL;
		cl_mem *p_inp_hdd2_lyr_wgt_conv_fp32 = NULL;
		cl_mem *p_inp_hdd2_lyr_wgt_bias_fp32 = NULL;

		cl_mem *p_ino_hdd2_lyr_data_fp32 = NULL;
		cl_mem *p_inp_hdd3_lyr_wgt_conv_fp32 = NULL;
		cl_mem *p_inp_hdd3_lyr_wgt_bias_fp32 = NULL;

		cl_mem *p_ino_hdd3_lyr_data_fp32 = NULL;

		cl_mem *p_out_label_s32 = NULL;
		cl_mem *p_out_conf_lv_fp32 = NULL;	

		/* event */
		cl_event *ev_write = NULL;
		cl_event *ev_kernel_i = NULL, *ev_kernel_1 = NULL, *ev_kernel_2 = NULL, *ev_kernel_3 = NULL, *ev_kernel_r = NULL;
		cl_event *ev_read = NULL;

		/* create buffer object */
		p_inp_lyr_data_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd0_lyr_wgt_conv_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd0_lyr_wgt_bias_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_ino_hdd0_lyr_data_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd1_lyr_wgt_conv_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd1_lyr_wgt_bias_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_ino_hdd1_lyr_data_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd2_lyr_wgt_conv_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd2_lyr_wgt_bias_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_ino_hdd2_lyr_data_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd3_lyr_wgt_conv_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd3_lyr_wgt_bias_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_ino_hdd3_lyr_data_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_out_label_s32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_out_conf_lv_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		for(i = 0; i < num_devs; i++)
		{
			p_inp_lyr_data_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * IMG_SIZE * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd0_lyr_wgt_conv_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * IMG_SIZE * size, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd0_lyr_wgt_bias_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size, NULL, &err);
			CHECK_ERROR(err);
			
			p_ino_hdd0_lyr_data_fp32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * size * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd1_lyr_wgt_conv_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size * size, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd1_lyr_wgt_bias_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size, NULL, &err);
			CHECK_ERROR(err);
			
			p_ino_hdd1_lyr_data_fp32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * size * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd2_lyr_wgt_conv_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size * size, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd2_lyr_wgt_bias_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size, NULL, &err);
			CHECK_ERROR(err);
			
			p_ino_hdd2_lyr_data_fp32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * size * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd3_lyr_wgt_conv_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size * DIGIT_COUNT, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd3_lyr_wgt_bias_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * DIGIT_COUNT, NULL, &err);
			CHECK_ERROR(err);
			
			p_ino_hdd3_lyr_data_fp32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * DIGIT_COUNT * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			
			p_out_label_s32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);		
			p_out_conf_lv_fp32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clCreateBuffer %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clCreateBuffer %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}

		ev_write = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_kernel_i = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_kernel_1 = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_kernel_2 = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_kernel_3 = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_kernel_r = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_read = (cl_event *)malloc(sizeof(cl_event) * num_devs);		
		
		/* write buffer */
		for(i = 0; i < num_devs; i++)
		{
#if (1 == PROFILING_ENABLE)
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_lyr_data_fp32[i], CL_FALSE, 0, sizeof(cl_float) * IMG_SIZE * IMG_COUNT / num_devs, &images[i * IMG_SIZE * IMG_COUNT / num_devs], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_lyr_data_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd0_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * IMG_SIZE * size, weights[0], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd0_lyr_wgt_conv_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd0_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[0], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd0_lyr_wgt_bias_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd1_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * size, weights[1], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd1_lyr_wgt_conv_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd1_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[1], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd1_lyr_wgt_bias_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd2_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * size, weights[2], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd2_lyr_wgt_conv_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd2_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[2], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd2_lyr_wgt_bias_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd3_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * DIGIT_COUNT, weights[3], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd3_lyr_wgt_conv_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd3_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * DIGIT_COUNT, biases[3], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd3_lyr_wgt_bias_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
#else
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_lyr_data_fp32[i], CL_FALSE, 0, sizeof(cl_float) * IMG_SIZE * IMG_COUNT / num_devs, &images[i * IMG_SIZE * IMG_COUNT / num_devs], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd0_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * IMG_SIZE * size, weights[0], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd0_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[0], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd1_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * size, weights[1], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd1_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[1], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd2_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * size, weights[2], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd2_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[2], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd3_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * DIGIT_COUNT, weights[3], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd3_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * DIGIT_COUNT, biases[3], 0, NULL, NULL);
			CHECK_ERROR(err);
#endif

#if (1 == DEBUGGING_INFO_PRINT)
			printf("clEnqueueWriteBuffer %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clEnqueueWriteBuffer %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}

#if (RUN_WITH_CL_CODE == RUN_MODE)
		/* create program object */
		for(i = 0; i < num_devs; i++)
		{
			program[i] = clCreateProgramWithSource(context, 1, (const char **)&code_kernel_src, &sz_kernel_src, &err);
			CHECK_ERROR(err);
		}
		
		free(code_kernel_src);
		
		/* build kernel source code */
		for(i = 0; i < num_devs; i++)
		{
#if (1 == CHECK_BUILD_ERROR_LOG)
			size_t log_size = 0;
			char *log = NULL;
#endif
			err = clBuildProgram(program[i], 1, &devs[i], "", NULL, NULL);
#if (1 == CHECK_BUILD_ERROR_LOG)
			clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			log = (char *)malloc(log_size + 1);
			clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
			log[log_size] = '\0';
			printf(":kernel Compile log:\n%s\n", log);
			free(log);
			log = NULL;
#endif
			CHECK_ERROR(err);
		}
#elif (PRE_BUILD_COMPILE == RUN_MODE)
		/* create program object */
		for(i = 0; i < num_devs; i++)
		{
			program[i] = clCreateProgramWithSource(context, 1, (const char **)&code_kernel_src, &sz_kernel_src, &err);
			CHECK_ERROR(err);
		}
		
		/* build kernel source code */
		for(i = 0; i < num_devs; i++)
		{
#if (1 == CHECK_BUILD_ERROR_LOG)
			size_t log_size = 0;
			char *log = NULL;
#endif
			err = clBuildProgram(program[i], 1, &devs[0], "", NULL, NULL);
#if (1 == CHECK_BUILD_ERROR_LOG)
			clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			log = (char *)malloc(log_size + 1);
			clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
			log[log_size] = '\0';
			printf(":kernel Compile log:\n%s\n", log);
			free(log);
			log = NULL;
#endif
			CHECK_ERROR(err);
		}
    
		size_t nbread;
		size_t *np = (size_t *)malloc(sizeof(size_t) * num_devs); /* Create size array */
		err = clGetProgramInfo(program[0], CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * num_devs, np, &nbread); /* Load in np the size of my binary */
		CHECK_ERROR(err);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("num_devs: %d nbread: %d np: %d %d %d %d\n", num_devs, nbread, np[0], np[1], np[2], np[3]);
#endif
		char** bn = (char **)malloc(sizeof(char *) * num_devs); /* Create the binary array */
		for(i = 0; i < num_devs; i++)
		{
			bn[i] = (char *)malloc(sizeof(char) * np[i]); /* I know... it's bad... but if i use new char[np[i]], i have a segfault... */ 
		}
		err = clGetProgramInfo(program[0], CL_PROGRAM_BINARIES, sizeof(unsigned char *)*num_devs, bn, &nbread); //Load the binary itself  
		CHECK_ERROR(err);
		FILE *fp = fopen(FILE_NAME_KERNEL_BIN_MEDIUM, "wb");
		i=0;
#if (1 == DEBUGGING_INFO_PRINT)
		printf("%s\n", bn[i]);
#endif
		fwrite(bn[i], sizeof(char), np[i], fp); // Save the binary, but my file stay empty  
		fclose(fp);  

		free(np);
		free(bn);

		printf("\n pre-build done !\n change PRE_BUILD_MODE mode as 0\n");
		exit(-1);
#else /* RUN_WITH_BINARY == RUN_MODE) */
		FILE *fp = fopen(FILE_NAME_KERNEL_BIN_MEDIUM, "rb");
		size_t binarySize[4];
		cl_int binaryStatus[4];
		fseek(fp, 0, SEEK_END);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("fseek done\n");
#endif
		binarySize[0] = ftell(fp);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("binarySize done: %d\n", binarySize[0]);
#endif
		rewind(fp);
		unsigned char *programBinary = (unsigned char *)malloc(binarySize[0]);
		fread(programBinary, 1, binarySize[0], fp);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("programBinary done: %d %d %d %d\n", programBinary[0], programBinary[1], programBinary[2], programBinary[3]);
#endif
		binarySize[0] = binarySize[0];
		binarySize[1] = binarySize[0];
		binarySize[2] = binarySize[0];
		binarySize[3] = binarySize[0];
#if (1 == DEBUGGING_INFO_PRINT)
		printf("binarySize: %d %d %d %d\n", binarySize[0], binarySize[1], binarySize[2], binarySize[3]);
#endif
		for(i = 0; i < num_devs; i++)
		{
			program[i] = clCreateProgramWithBinary(context, 1, &devs[i], &binarySize[i], (const unsigned char**)&programBinary, &binaryStatus[i], &err);
			CHECK_ERROR(err);			
		}
#if (1 == DEBUGGING_INFO_PRINT)
		printf("clCreateProgramWithBinary done: %u %u %u %u\n", binaryStatus[0], binaryStatus[1], binaryStatus[2], binaryStatus[3]);
#endif
		CHECK_ERROR(err);
		fclose(fp);

		/* build kernel source code */
		for(i = 0; i < num_devs; i++)
		{
			err = clBuildProgram(program[i], 1, &devs[i], "", NULL, NULL);
			CHECK_ERROR(err);
		}
#if (1 == DEBUGGING_INFO_PRINT)
		printf("clBuildProgram done\n");
#endif
#if (1 == CHECK_BUILD_ERROR_LOG)
		{
			size_t log_size = 0;
			char *log = NULL;
			for(i = 0; i < num_devs; i++)
			{
				clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
				log = (char *)malloc(log_size + 1);
				clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
				log[log_size] = '\0';
				printf(":kernel Compile log:\n%s\n", log);
				free(log);
				log = NULL;
			}
		}
#endif
		CHECK_ERROR(err);
#endif 

#if (1 == DEBUGGING_INFO_PRINT)
		printf("clBuildProgram done\n");
#if (1 == PROFILING_ENABLE)
		clock_gettime(CLOCK_MONOTONIC, &end);
		timespec_subtract(&spent, &end, &start);
		printf("clBuildProgram time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
#endif

		/* create kernel object */
		kernel_inp_lyr = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devs);
		kernel_hdd1_lyr = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devs);
		kernel_hdd2_lyr = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devs);
		kernel_out_lyr = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devs);
		kernel_red_lyr = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devs);
		for(i = 0; i < num_devs; i++)
		{
			kernel_inp_lyr[i] = clCreateKernel(program[i], "kernel_med_inp_lyr", &err);
			CHECK_ERROR(err);			
			kernel_hdd1_lyr[i] = clCreateKernel(program[i], "kernel_med_hdn_lyr", &err);
			CHECK_ERROR(err);
			kernel_hdd2_lyr[i] = clCreateKernel(program[i], "kernel_med_hdn_lyr", &err);
			CHECK_ERROR(err);
			kernel_out_lyr[i] = clCreateKernel(program[i], "kernel_med_out_lyr", &err);
			CHECK_ERROR(err);
			kernel_red_lyr[i] = clCreateKernel(program[i], "kernel_reduction_lyr", &err);
			CHECK_ERROR(err);
			
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clCreateKernel %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clCreateKernel %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}

		/* kernel argument setting */
		for(i = 0; i < num_devs; i++)
		{
			err = clSetKernelArg(kernel_inp_lyr[i], 0, sizeof(cl_mem), &p_inp_lyr_data_fp32[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(kernel_inp_lyr[i], 1, sizeof(cl_mem), &p_inp_hdd0_lyr_wgt_conv_fp32[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(kernel_inp_lyr[i], 2, sizeof(cl_mem), &p_inp_hdd0_lyr_wgt_bias_fp32[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(kernel_inp_lyr[i], 3, sizeof(cl_mem), &p_ino_hdd0_lyr_data_fp32[i]);
			CHECK_ERROR(err);

			err = clSetKernelArg(kernel_hdd1_lyr[i], 0, sizeof(cl_mem), &p_ino_hdd0_lyr_data_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd1_lyr[i], 1, sizeof(cl_mem), &p_inp_hdd1_lyr_wgt_conv_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd1_lyr[i], 2, sizeof(cl_mem), &p_inp_hdd1_lyr_wgt_bias_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd1_lyr[i], 3, sizeof(cl_mem), &p_ino_hdd1_lyr_data_fp32[i]);
			CHECK_ERROR(err);

			err = clSetKernelArg(kernel_hdd2_lyr[i], 0, sizeof(cl_mem), &p_ino_hdd1_lyr_data_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd2_lyr[i], 1, sizeof(cl_mem), &p_inp_hdd2_lyr_wgt_conv_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd2_lyr[i], 2, sizeof(cl_mem), &p_inp_hdd2_lyr_wgt_bias_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd2_lyr[i], 3, sizeof(cl_mem), &p_ino_hdd2_lyr_data_fp32[i]);
			CHECK_ERROR(err);

			err = clSetKernelArg(kernel_out_lyr[i], 0, sizeof(cl_mem), &p_ino_hdd2_lyr_data_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_out_lyr[i], 1, sizeof(cl_mem), &p_inp_hdd3_lyr_wgt_conv_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_out_lyr[i], 2, sizeof(cl_mem), &p_inp_hdd3_lyr_wgt_bias_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_out_lyr[i], 3, sizeof(cl_mem), &p_ino_hdd3_lyr_data_fp32[i]);
			CHECK_ERROR(err);

			err = clSetKernelArg(kernel_red_lyr[i], 0, sizeof(cl_mem), &p_ino_hdd3_lyr_data_fp32[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(kernel_red_lyr[i], 1, sizeof(cl_mem), &p_out_label_s32[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(kernel_red_lyr[i], 2, sizeof(cl_mem), &p_out_conf_lv_fp32[i]);
			CHECK_ERROR(err);
			
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clSetKernelArg %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clSetKernelArg %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}
		
		/* run kernel */
		for(i = 0; i < num_devs; i++)
		{
			err = clEnqueueNDRangeKernel(cmd_queues[i], kernel_inp_lyr[i], 1, NULL, &sz_global_recognition, &sz_local_recognition, 0, NULL, &ev_kernel_i[i]);
			CHECK_ERROR(err);
			err = clEnqueueNDRangeKernel(cmd_queues[i], kernel_hdd1_lyr[i], 1, NULL, &sz_global_recognition, &sz_local_recognition, 0, NULL, &ev_kernel_1[i]);
			CHECK_ERROR(err);
			err = clEnqueueNDRangeKernel(cmd_queues[i], kernel_hdd2_lyr[i], 1, NULL, &sz_global_recognition, &sz_local_recognition, 0, NULL, &ev_kernel_2[i]);
			CHECK_ERROR(err);
			err = clEnqueueNDRangeKernel(cmd_queues[i], kernel_out_lyr[i], 1, NULL, &sz_global_recognition, &sz_local_recognition, 0, NULL, &ev_kernel_3[i]);
			CHECK_ERROR(err);

			err = clEnqueueNDRangeKernel(cmd_queues[i], kernel_red_lyr[i], 1, NULL, &sz_global_recognition, &sz_local_recognition, 0, NULL, &ev_kernel_r[i]);
			CHECK_ERROR(err);

#if (1 == DEBUGGING_INFO_PRINT)
			printf("kernel %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("kernel %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif			
		}

		/* read buffer */
		for(i = 0; i < num_devs; i++)
		{
#if (1 == PROFILING_ENABLE)
			err = clEnqueueReadBuffer(cmd_queues[i], p_out_label_s32[i], CL_TRUE, 0, sizeof(cl_int) * IMG_COUNT / num_devs, &labels[i * IMG_COUNT / num_devs], 1, &ev_kernel_r[i], &ev_read[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueReadBuffer p_out_label_s32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueReadBuffer(cmd_queues[i], p_out_conf_lv_fp32[i], CL_TRUE, 0, sizeof(cl_float) * IMG_COUNT / num_devs, &confidences[i * IMG_COUNT / num_devs], 1, &ev_kernel_r[i], &ev_read[i]);
			CHECK_ERROR(err);			
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueReadBuffer p_out_conf_lv_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
#else
			err = clEnqueueReadBuffer(cmd_queues[i], p_out_label_s32[i], CL_TRUE, 0, sizeof(cl_int) * IMG_COUNT / num_devs, &labels[i * IMG_COUNT / num_devs], 1, &ev_kernel_r[i], NULL);
			CHECK_ERROR(err);
			err = clEnqueueReadBuffer(cmd_queues[i], p_out_conf_lv_fp32[i], CL_TRUE, 0, sizeof(cl_float) * IMG_COUNT / num_devs, &confidences[i * IMG_COUNT / num_devs], 1, &ev_kernel_r[i], NULL);
			CHECK_ERROR(err);			
#endif

#if (1 == DEBUGGING_INFO_PRINT)
			printf("clEnqueueReadBuffer %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clEnqueueReadBuffer %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}

#if (1 == PROFILING_ENABLE)
		/* kernel profile results print */
		for(i = 0; i < num_devs; i++)
		{
			err = clGetEventProfilingInfo(ev_kernel_i[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_i[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_i[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_i[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueNDRangeKernel kernel_inp_lyr %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clGetEventProfilingInfo(ev_kernel_1[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_1[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_1[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_1[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueNDRangeKernel kernel_hdd1_lyr %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clGetEventProfilingInfo(ev_kernel_2[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_2[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_2[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_2[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueNDRangeKernel kernel_hdd2_lyr %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clGetEventProfilingInfo(ev_kernel_3[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_3[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_3[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_3[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueNDRangeKernel kernel_out_lyr %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);			
			err = clGetEventProfilingInfo(ev_kernel_r[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_r[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_r[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_r[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueNDRangeKernel kernel_red_lyr %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);			
			
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clGetEventProfilingInfo %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clGetEventProfilingInfo %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}
#endif		
		
		/* release stage */
		for(i = 0; i < num_devs; i++)
		{
			clReleaseMemObject(p_inp_lyr_data_fp32[i]);
			clReleaseMemObject(p_inp_hdd0_lyr_wgt_conv_fp32[i]);
			clReleaseMemObject(p_inp_hdd0_lyr_wgt_bias_fp32[i]);
			clReleaseMemObject(p_ino_hdd0_lyr_data_fp32[i]);
			clReleaseMemObject(p_inp_hdd1_lyr_wgt_conv_fp32[i]);
			clReleaseMemObject(p_inp_hdd1_lyr_wgt_bias_fp32[i]);
			clReleaseMemObject(p_ino_hdd1_lyr_data_fp32[i]);
			clReleaseMemObject(p_inp_hdd2_lyr_wgt_conv_fp32[i]);
			clReleaseMemObject(p_inp_hdd2_lyr_wgt_bias_fp32[i]);
			clReleaseMemObject(p_ino_hdd2_lyr_data_fp32[i]);
			clReleaseMemObject(p_inp_hdd3_lyr_wgt_conv_fp32[i]);
			clReleaseMemObject(p_inp_hdd3_lyr_wgt_bias_fp32[i]);
			clReleaseMemObject(p_ino_hdd3_lyr_data_fp32[i]);
			clReleaseMemObject(p_out_label_s32[i]);
			clReleaseMemObject(p_out_conf_lv_fp32[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseMemObject %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseMemObject %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseKernel(kernel_inp_lyr[i]);
			clReleaseKernel(kernel_hdd1_lyr[i]);
			clReleaseKernel(kernel_hdd2_lyr[i]);
			clReleaseKernel(kernel_out_lyr[i]);
			clReleaseKernel(kernel_red_lyr[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseKernel %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseKernel %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
#if (1 == PROFILING_ENABLE)
			clReleaseEvent(ev_write[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_write[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_write[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
#endif
#if (1 == PROFILING_ENABLE)
			clReleaseEvent(ev_kernel_i[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_kernel_0[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_kernel_0[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseEvent(ev_kernel_1[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_kernel_1[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_kernel_1[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseEvent(ev_kernel_2[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_kernel_2[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_kernel_2[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		clReleaseEvent(ev_kernel_3[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_kernel_3[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_kernel_3[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseEvent(ev_kernel_r[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_kernel_r[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_kernel_r[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseEvent(ev_read[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_read[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_read[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif		
#endif
			clReleaseProgram(program[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseProgram %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseProgram %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}

		free(kernel_inp_lyr);
		free(kernel_hdd1_lyr);
		free(kernel_hdd2_lyr);
		free(kernel_out_lyr);
		free(kernel_red_lyr);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("free kernel done\n");
#endif
#if (1 == PROFILING_ENABLE)
		clock_gettime(CLOCK_MONOTONIC, &end);
		timespec_subtract(&spent, &end, &start);
		printf("free kernel time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		free(p_inp_lyr_data_fp32);
		free(p_inp_hdd0_lyr_wgt_conv_fp32);
		free(p_inp_hdd0_lyr_wgt_bias_fp32);
		free(p_ino_hdd0_lyr_data_fp32);
		free(p_inp_hdd1_lyr_wgt_conv_fp32);
		free(p_inp_hdd1_lyr_wgt_bias_fp32);
		free(p_ino_hdd1_lyr_data_fp32);
		free(p_inp_hdd2_lyr_wgt_conv_fp32);
		free(p_inp_hdd2_lyr_wgt_bias_fp32);
		free(p_ino_hdd2_lyr_data_fp32);
		free(p_inp_hdd3_lyr_wgt_conv_fp32);
		free(p_inp_hdd3_lyr_wgt_bias_fp32);
		free(p_ino_hdd3_lyr_data_fp32);
		free(p_out_label_s32);
		free(p_out_conf_lv_fp32);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("free mem done\n");
#endif
#if (1 == PROFILING_ENABLE)
		clock_gettime(CLOCK_MONOTONIC, &end);
		timespec_subtract(&spent, &end, &start);
		printf("free mem time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
	}
	else if(size == 1024) /* large network */
	{
#if (RUN_WITH_CL_CODE == RUN_MODE)
		size_t sz_kernel_src;
		char *code_kernel_src = get_source_code(FILE_NAME_KERNEL_CODE_LARGE, &sz_kernel_src);
#elif (PRE_BUILD_COMPILE == RUN_MODE)
		size_t sz_kernel_src;
		char *code_kernel_src = get_source_code(FILE_NAME_KERNEL_CODE_LARGE, &sz_kernel_src);
#else /* RUN_WITH_BINARY == RUN_MODE) */
#endif

		/* kernel object */
		cl_kernel *kernel_inp_lyr  = NULL;
		cl_kernel *kernel_hdd1_lyr  = NULL;
		cl_kernel *kernel_hdd2_lyr  = NULL;
		cl_kernel *kernel_hdd3_lyr  = NULL;
		cl_kernel *kernel_out_lyr  = NULL;
		cl_kernel *kernel_red_lyr  = NULL;

		/* memory object */
		cl_mem *p_inp_lyr_data_fp32 = NULL;
		cl_mem *p_inp_hdd0_lyr_wgt_conv_fp32 = NULL;
		cl_mem *p_inp_hdd0_lyr_wgt_bias_fp32 = NULL;

		cl_mem *p_ino_hdd0_lyr_data_fp32 = NULL;
		cl_mem *p_inp_hdd1_lyr_wgt_conv_fp32 = NULL;
		cl_mem *p_inp_hdd1_lyr_wgt_bias_fp32 = NULL;

		cl_mem *p_ino_hdd1_lyr_data_fp32 = NULL;
		cl_mem *p_inp_hdd2_lyr_wgt_conv_fp32 = NULL;
		cl_mem *p_inp_hdd2_lyr_wgt_bias_fp32 = NULL;

		cl_mem *p_ino_hdd2_lyr_data_fp32 = NULL;
		cl_mem *p_inp_hdd3_lyr_wgt_conv_fp32 = NULL;
		cl_mem *p_inp_hdd3_lyr_wgt_bias_fp32 = NULL;

		cl_mem *p_ino_hdd3_lyr_data_fp32 = NULL;
		cl_mem *p_inp_hdd4_lyr_wgt_conv_fp32 = NULL;
		cl_mem *p_inp_hdd4_lyr_wgt_bias_fp32 = NULL;
		cl_mem *p_ino_hdd4_lyr_data_fp32 = NULL;

		cl_mem *p_out_label_s32 = NULL;
		cl_mem *p_out_conf_lv_fp32 = NULL;	

		/* event */
		cl_event *ev_write = NULL;
		cl_event *ev_kernel_i = NULL, *ev_kernel_1 = NULL, *ev_kernel_2 = NULL, *ev_kernel_3 = NULL, *ev_kernel_4 = NULL, *ev_kernel_r = NULL;
		cl_event *ev_read = NULL;

		/* create buffer object */
		p_inp_lyr_data_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd0_lyr_wgt_conv_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd0_lyr_wgt_bias_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_ino_hdd0_lyr_data_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd1_lyr_wgt_conv_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd1_lyr_wgt_bias_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_ino_hdd1_lyr_data_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd2_lyr_wgt_conv_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd2_lyr_wgt_bias_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_ino_hdd2_lyr_data_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd3_lyr_wgt_conv_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd3_lyr_wgt_bias_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_ino_hdd3_lyr_data_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd4_lyr_wgt_conv_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_inp_hdd4_lyr_wgt_bias_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_ino_hdd4_lyr_data_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_out_label_s32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		p_out_conf_lv_fp32 = (cl_mem *)malloc(sizeof(cl_mem) * num_devs);
		for(i = 0; i < num_devs; i++)
		{
			p_inp_lyr_data_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * IMG_SIZE * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd0_lyr_wgt_conv_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * IMG_SIZE * size, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd0_lyr_wgt_bias_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size, NULL, &err);
			CHECK_ERROR(err);
			
			p_ino_hdd0_lyr_data_fp32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * size * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd1_lyr_wgt_conv_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size * size, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd1_lyr_wgt_bias_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size, NULL, &err);
			CHECK_ERROR(err);
			
			p_ino_hdd1_lyr_data_fp32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * size * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd2_lyr_wgt_conv_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size * size, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd2_lyr_wgt_bias_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size, NULL, &err);
			CHECK_ERROR(err);
			
			p_ino_hdd2_lyr_data_fp32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * size * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd3_lyr_wgt_conv_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size * size, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd3_lyr_wgt_bias_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size, NULL, &err);
			CHECK_ERROR(err);
			
			p_ino_hdd3_lyr_data_fp32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * size * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd4_lyr_wgt_conv_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * size * DIGIT_COUNT, NULL, &err);
			CHECK_ERROR(err);
			p_inp_hdd4_lyr_wgt_bias_fp32[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * DIGIT_COUNT, NULL, &err);
			CHECK_ERROR(err);
			
			p_ino_hdd4_lyr_data_fp32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * DIGIT_COUNT * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			
			p_out_label_s32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);		
			p_out_conf_lv_fp32[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * IMG_COUNT / num_devs, NULL, &err);
			CHECK_ERROR(err);
			
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clCreateBuffer %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clCreateBuffer %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}

		ev_write = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_kernel_i = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_kernel_1 = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_kernel_2 = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_kernel_3 = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_kernel_4 = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_kernel_r = (cl_event *)malloc(sizeof(cl_event) * num_devs);
		ev_read = (cl_event *)malloc(sizeof(cl_event) * num_devs);		
		
		/* write buffer */
		for(i = 0; i < num_devs; i++)
		{
#if (1 == PROFILING_ENABLE)
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_lyr_data_fp32[i], CL_FALSE, 0, sizeof(cl_float) * IMG_SIZE * IMG_COUNT / num_devs, &images[i * IMG_SIZE * IMG_COUNT / num_devs], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_lyr_data_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd0_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * IMG_SIZE * size, weights[0], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd0_lyr_wgt_conv_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd0_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[0], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd0_lyr_wgt_bias_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd1_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * size, weights[1], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd1_lyr_wgt_conv_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd1_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[1], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd1_lyr_wgt_bias_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd2_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * size, weights[2], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd2_lyr_wgt_conv_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd2_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[2], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd2_lyr_wgt_bias_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd3_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * size, weights[3], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd3_lyr_wgt_conv_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd3_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[3], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd3_lyr_wgt_bias_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd4_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * DIGIT_COUNT, weights[4], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd4_lyr_wgt_conv_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd4_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * DIGIT_COUNT, biases[4], 0, NULL, &ev_write[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_write[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueWriteBuffer p_inp_hdd4_lyr_wgt_bias_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
#else
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_lyr_data_fp32[i], CL_FALSE, 0, sizeof(cl_float) * IMG_SIZE * IMG_COUNT / num_devs, &images[i * IMG_SIZE * IMG_COUNT / num_devs], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd0_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * IMG_SIZE * size, weights[0], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd0_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[0], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd1_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * size, weights[1], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd1_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[1], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd2_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * size, weights[2], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd2_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[2], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd3_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * size, weights[3], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd3_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size, biases[3], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd4_lyr_wgt_conv_fp32[i], CL_FALSE, 0, sizeof(cl_float) * size * DIGIT_COUNT, weights[4], 0, NULL, NULL);
			CHECK_ERROR(err);
			err = clEnqueueWriteBuffer(cmd_queues[i], p_inp_hdd4_lyr_wgt_bias_fp32[i], CL_FALSE, 0, sizeof(cl_float) * DIGIT_COUNT, biases[4], 0, NULL, NULL);
			CHECK_ERROR(err);
#endif

#if (1 == DEBUGGING_INFO_PRINT)
			printf("clEnqueueWriteBuffer %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clEnqueueWriteBuffer %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}

#if (RUN_WITH_CL_CODE == RUN_MODE)
		/* create program object */
		for(i = 0; i < num_devs; i++)
		{
			program[i] = clCreateProgramWithSource(context, 1, (const char **)&code_kernel_src, &sz_kernel_src, &err);
			CHECK_ERROR(err);
		}
		
		free(code_kernel_src);
		
		/* build kernel source code */
		for(i = 0; i < num_devs; i++)
		{
#if (1 == CHECK_BUILD_ERROR_LOG)
			size_t log_size = 0;
			char *log = NULL;
#endif
			err = clBuildProgram(program[i], 1, &devs[i], "", NULL, NULL);
#if (1 == CHECK_BUILD_ERROR_LOG)
			clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			log = (char *)malloc(log_size + 1);
			clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
			log[log_size] = '\0';
			printf(":kernel Compile log:\n%s\n", log);
			free(log);
			log = NULL;
#endif
			CHECK_ERROR(err);
		}
#elif (PRE_BUILD_COMPILE == RUN_MODE)
		/* create program object */
		for(i = 0; i < num_devs; i++)
		{
			program[i] = clCreateProgramWithSource(context, 1, (const char **)&code_kernel_src, &sz_kernel_src, &err);
			CHECK_ERROR(err);
		}
		
		/* build kernel source code */
		for(i = 0; i < num_devs; i++)
		{
#if (1 == CHECK_BUILD_ERROR_LOG)
			size_t log_size = 0;
			char *log = NULL;
#endif
			err = clBuildProgram(program[i], 1, &devs[0], "", NULL, NULL);
#if (1 == CHECK_BUILD_ERROR_LOG)
			clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			log = (char *)malloc(log_size + 1);
			clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
			log[log_size] = '\0';
			printf(":kernel Compile log:\n%s\n", log);
			free(log);
			log = NULL;
#endif
			CHECK_ERROR(err);
		}
    
		size_t nbread;
		size_t *np = (size_t *)malloc(sizeof(size_t) * num_devs); /* Create size array */
		err = clGetProgramInfo(program[0], CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * num_devs, np, &nbread); /* Load in np the size of my binary */
		CHECK_ERROR(err);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("num_devs: %d nbread: %d np: %d %d %d %d\n", num_devs, nbread, np[0], np[1], np[2], np[3]);
#endif
		char** bn = (char **)malloc(sizeof(char *) * num_devs); /* Create the binary array */
		for(i = 0; i < num_devs; i++)
		{
			bn[i] = (char *)malloc(sizeof(char) * np[i]); /* I know... it's bad... but if i use new char[np[i]], i have a segfault... */ 
		}
		err = clGetProgramInfo(program[0], CL_PROGRAM_BINARIES, sizeof(unsigned char *)*num_devs, bn, &nbread); //Load the binary itself  
		CHECK_ERROR(err);
		FILE *fp = fopen(FILE_NAME_KERNEL_BIN_LARGE, "wb");
		i=0;
#if (1 == DEBUGGING_INFO_PRINT)
		printf("%s\n", bn[i]);
#endif
		fwrite(bn[i], sizeof(char), np[i], fp); // Save the binary, but my file stay empty  
		fclose(fp);  

		free(np);
		free(bn);

		printf("\n pre-build done !\n change PRE_BUILD_MODE mode as 0\n");
		exit(-1);
#else /* RUN_WITH_BINARY == RUN_MODE) */
		FILE *fp = fopen(FILE_NAME_KERNEL_BIN_LARGE, "rb");
		size_t binarySize[4];
		cl_int binaryStatus[4];
		fseek(fp, 0, SEEK_END);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("fseek done\n");
#endif
		binarySize[0] = ftell(fp);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("binarySize done: %d\n", binarySize[0]);
#endif
		rewind(fp);
		unsigned char *programBinary = (unsigned char *)malloc(binarySize[0]);
		fread(programBinary, 1, binarySize[0], fp);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("programBinary done: %d %d %d %d\n", programBinary[0], programBinary[1], programBinary[2], programBinary[3]);
#endif
		binarySize[0] = binarySize[0];
		binarySize[1] = binarySize[0];
		binarySize[2] = binarySize[0];
		binarySize[3] = binarySize[0];
#if (1 == DEBUGGING_INFO_PRINT)
		printf("binarySize: %d %d %d %d\n", binarySize[0], binarySize[1], binarySize[2], binarySize[3]);
#endif
		for(i = 0; i < num_devs; i++)
		{
			program[i] = clCreateProgramWithBinary(context, 1, &devs[i], &binarySize[i], (const unsigned char**)&programBinary, &binaryStatus[i], &err);
			CHECK_ERROR(err);			
		}
#if (1 == DEBUGGING_INFO_PRINT)
		printf("clCreateProgramWithBinary done: %u %u %u %u\n", binaryStatus[0], binaryStatus[1], binaryStatus[2], binaryStatus[3]);
#endif
		CHECK_ERROR(err);
		fclose(fp);

		/* build kernel source code */
		for(i = 0; i < num_devs; i++)
		{
			err = clBuildProgram(program[i], 1, &devs[i], "", NULL, NULL);
			CHECK_ERROR(err);
		}
#if (1 == DEBUGGING_INFO_PRINT)
		printf("clBuildProgram done\n");
#endif
#if (1 == CHECK_BUILD_ERROR_LOG)
		{
			size_t log_size = 0;
			char *log = NULL;
			for(i = 0; i < num_devs; i++)
			{
				clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
				log = (char *)malloc(log_size + 1);
				clGetProgramBuildInfo(program[i], devs[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
				log[log_size] = '\0';
				printf(":kernel Compile log:\n%s\n", log);
				free(log);
				log = NULL;
			}
		}
#endif
		CHECK_ERROR(err);
#endif 

#if (1 == DEBUGGING_INFO_PRINT)
		printf("clBuildProgram done\n");
#if (1 == PROFILING_ENABLE)
		clock_gettime(CLOCK_MONOTONIC, &end);
		timespec_subtract(&spent, &end, &start);
		printf("clBuildProgram time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
#endif

		/* create kernel object */
		kernel_inp_lyr = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devs);
		kernel_hdd1_lyr = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devs);
		kernel_hdd2_lyr = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devs);
		kernel_hdd3_lyr = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devs);
		kernel_out_lyr = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devs);
		kernel_red_lyr = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devs);
		for(i = 0; i < num_devs; i++)
		{
			kernel_inp_lyr[i] = clCreateKernel(program[i], "kernel_lrg_inp_lyr", &err);
			CHECK_ERROR(err);			
			kernel_hdd1_lyr[i] = clCreateKernel(program[i], "kernel_lrg_hdn_lyr", &err);
			CHECK_ERROR(err);
			kernel_hdd2_lyr[i] = clCreateKernel(program[i], "kernel_lrg_hdn_lyr", &err);
			CHECK_ERROR(err);
			kernel_hdd3_lyr[i] = clCreateKernel(program[i], "kernel_lrg_hdn_lyr", &err);
			CHECK_ERROR(err);
			kernel_out_lyr[i] = clCreateKernel(program[i], "kernel_lrg_out_lyr", &err);
			CHECK_ERROR(err);
			kernel_red_lyr[i] = clCreateKernel(program[i], "kernel_reduction_lyr", &err);
			CHECK_ERROR(err);
			
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clCreateKernel %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clCreateKernel %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}

		/* kernel argument setting */
		for(i = 0; i < num_devs; i++)
		{
			err = clSetKernelArg(kernel_inp_lyr[i], 0, sizeof(cl_mem), &p_inp_lyr_data_fp32[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(kernel_inp_lyr[i], 1, sizeof(cl_mem), &p_inp_hdd0_lyr_wgt_conv_fp32[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(kernel_inp_lyr[i], 2, sizeof(cl_mem), &p_inp_hdd0_lyr_wgt_bias_fp32[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(kernel_inp_lyr[i], 3, sizeof(cl_mem), &p_ino_hdd0_lyr_data_fp32[i]);
			CHECK_ERROR(err);

			err = clSetKernelArg(kernel_hdd1_lyr[i], 0, sizeof(cl_mem), &p_ino_hdd0_lyr_data_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd1_lyr[i], 1, sizeof(cl_mem), &p_inp_hdd1_lyr_wgt_conv_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd1_lyr[i], 2, sizeof(cl_mem), &p_inp_hdd1_lyr_wgt_bias_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd1_lyr[i], 3, sizeof(cl_mem), &p_ino_hdd1_lyr_data_fp32[i]);
			CHECK_ERROR(err);

			err = clSetKernelArg(kernel_hdd2_lyr[i], 0, sizeof(cl_mem), &p_ino_hdd1_lyr_data_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd2_lyr[i], 1, sizeof(cl_mem), &p_inp_hdd2_lyr_wgt_conv_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd2_lyr[i], 2, sizeof(cl_mem), &p_inp_hdd2_lyr_wgt_bias_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd2_lyr[i], 3, sizeof(cl_mem), &p_ino_hdd2_lyr_data_fp32[i]);
			CHECK_ERROR(err);

			err = clSetKernelArg(kernel_hdd3_lyr[i], 0, sizeof(cl_mem), &p_ino_hdd2_lyr_data_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd3_lyr[i], 1, sizeof(cl_mem), &p_inp_hdd3_lyr_wgt_conv_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd3_lyr[i], 2, sizeof(cl_mem), &p_inp_hdd3_lyr_wgt_bias_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_hdd3_lyr[i], 3, sizeof(cl_mem), &p_ino_hdd3_lyr_data_fp32[i]);
			CHECK_ERROR(err);

			err = clSetKernelArg(kernel_out_lyr[i], 0, sizeof(cl_mem), &p_ino_hdd3_lyr_data_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_out_lyr[i], 1, sizeof(cl_mem), &p_inp_hdd4_lyr_wgt_conv_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_out_lyr[i], 2, sizeof(cl_mem), &p_inp_hdd4_lyr_wgt_bias_fp32[i]);
			CHECK_ERROR(err);                            
			err = clSetKernelArg(kernel_out_lyr[i], 3, sizeof(cl_mem), &p_ino_hdd4_lyr_data_fp32[i]);
			CHECK_ERROR(err);

			err = clSetKernelArg(kernel_red_lyr[i], 0, sizeof(cl_mem), &p_ino_hdd4_lyr_data_fp32[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(kernel_red_lyr[i], 1, sizeof(cl_mem), &p_out_label_s32[i]);
			CHECK_ERROR(err);
			err = clSetKernelArg(kernel_red_lyr[i], 2, sizeof(cl_mem), &p_out_conf_lv_fp32[i]);
			CHECK_ERROR(err);
			
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clSetKernelArg %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clSetKernelArg %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}
		
		/* run kernel */
		for(i = 0; i < num_devs; i++)
		{
			err = clEnqueueNDRangeKernel(cmd_queues[i], kernel_inp_lyr[i], 1, NULL, &sz_global_recognition, &sz_local_recognition, 0, NULL, &ev_kernel_i[i]);
			CHECK_ERROR(err);
			err = clEnqueueNDRangeKernel(cmd_queues[i], kernel_hdd1_lyr[i], 1, NULL, &sz_global_recognition, &sz_local_recognition, 0, NULL, &ev_kernel_1[i]);
			CHECK_ERROR(err);
			err = clEnqueueNDRangeKernel(cmd_queues[i], kernel_hdd2_lyr[i], 1, NULL, &sz_global_recognition, &sz_local_recognition, 0, NULL, &ev_kernel_2[i]);
			CHECK_ERROR(err);
			err = clEnqueueNDRangeKernel(cmd_queues[i], kernel_hdd3_lyr[i], 1, NULL, &sz_global_recognition, &sz_local_recognition, 0, NULL, &ev_kernel_3[i]);
			CHECK_ERROR(err);
			err = clEnqueueNDRangeKernel(cmd_queues[i], kernel_out_lyr[i], 1, NULL, &sz_global_recognition, &sz_local_recognition, 0, NULL, &ev_kernel_4[i]);
			CHECK_ERROR(err);

			err = clEnqueueNDRangeKernel(cmd_queues[i], kernel_red_lyr[i], 1, NULL, &sz_global_recognition, &sz_local_recognition, 0, NULL, &ev_kernel_r[i]);
			CHECK_ERROR(err);

#if (1 == DEBUGGING_INFO_PRINT)
			printf("kernel %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("kernel %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif			
		}

		/* read buffer */
		for(i = 0; i < num_devs; i++)
		{
#if (1 == PROFILING_ENABLE)
			err = clEnqueueReadBuffer(cmd_queues[i], p_out_label_s32[i], CL_TRUE, 0, sizeof(cl_int) * IMG_COUNT / num_devs, &labels[i * IMG_COUNT / num_devs], 1, &ev_kernel_r[i], &ev_read[i]);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueReadBuffer p_out_label_s32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clEnqueueReadBuffer(cmd_queues[i], p_out_conf_lv_fp32[i], CL_TRUE, 0, sizeof(cl_float) * IMG_COUNT / num_devs, &confidences[i * IMG_COUNT / num_devs], 1, &ev_kernel_r[i], &ev_read[i]);
			CHECK_ERROR(err);			
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_read[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueReadBuffer p_out_conf_lv_fp32 %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
#else
			err = clEnqueueReadBuffer(cmd_queues[i], p_out_label_s32[i], CL_TRUE, 0, sizeof(cl_int) * IMG_COUNT / num_devs, &labels[i * IMG_COUNT / num_devs], 1, &ev_kernel_r[i], NULL);
			CHECK_ERROR(err);
			err = clEnqueueReadBuffer(cmd_queues[i], p_out_conf_lv_fp32[i], CL_TRUE, 0, sizeof(cl_float) * IMG_COUNT / num_devs, &confidences[i * IMG_COUNT / num_devs], 1, &ev_kernel_r[i], NULL);
			CHECK_ERROR(err);			
#endif

#if (1 == DEBUGGING_INFO_PRINT)
			printf("clEnqueueReadBuffer %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clEnqueueReadBuffer %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}

#if (1 == PROFILING_ENABLE)
		/* kernel profile results print */
		for(i = 0; i < num_devs; i++)
		{
			err = clGetEventProfilingInfo(ev_kernel_i[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_i[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_i[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_i[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueNDRangeKernel kernel_inp_lyr %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clGetEventProfilingInfo(ev_kernel_1[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_1[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_1[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_1[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueNDRangeKernel kernel_hdd1_lyr %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clGetEventProfilingInfo(ev_kernel_2[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_2[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_2[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_2[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueNDRangeKernel kernel_hdd2_lyr %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);
			err = clGetEventProfilingInfo(ev_kernel_3[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_3[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_3[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_3[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueNDRangeKernel kernel_hdd3_lyr %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);			
			err = clGetEventProfilingInfo(ev_kernel_4[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_4[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_4[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_4[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueNDRangeKernel kernel_out_lyr %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);			
			err = clGetEventProfilingInfo(ev_kernel_r[i], CL_PROFILING_COMMAND_QUEUED, sizeof(queued_time), &queued_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_r[i], CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_r[i], CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
			CHECK_ERROR(err);
			err = clGetEventProfilingInfo(ev_kernel_r[i], CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
			CHECK_ERROR(err);
			printf("clEnqueueNDRangeKernel kernel_red_lyr %d: %lu %lu %lu %lu %lu ns\n", i, queued_time, submit_time, start_time, end_time, end_time - start_time);			
			
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clGetEventProfilingInfo %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clGetEventProfilingInfo %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}
#endif		
		
		/* release stage */
		for(i = 0; i < num_devs; i++)
		{
			clReleaseMemObject(p_inp_lyr_data_fp32[i]);
			clReleaseMemObject(p_inp_hdd0_lyr_wgt_conv_fp32[i]);
			clReleaseMemObject(p_inp_hdd0_lyr_wgt_bias_fp32[i]);
			clReleaseMemObject(p_ino_hdd0_lyr_data_fp32[i]);
			clReleaseMemObject(p_inp_hdd1_lyr_wgt_conv_fp32[i]);
			clReleaseMemObject(p_inp_hdd1_lyr_wgt_bias_fp32[i]);
			clReleaseMemObject(p_ino_hdd1_lyr_data_fp32[i]);
			clReleaseMemObject(p_inp_hdd2_lyr_wgt_conv_fp32[i]);
			clReleaseMemObject(p_inp_hdd2_lyr_wgt_bias_fp32[i]);
			clReleaseMemObject(p_ino_hdd2_lyr_data_fp32[i]);
			clReleaseMemObject(p_inp_hdd3_lyr_wgt_conv_fp32[i]);
			clReleaseMemObject(p_inp_hdd3_lyr_wgt_bias_fp32[i]);
			clReleaseMemObject(p_ino_hdd3_lyr_data_fp32[i]);
			clReleaseMemObject(p_inp_hdd4_lyr_wgt_conv_fp32[i]);
			clReleaseMemObject(p_inp_hdd4_lyr_wgt_bias_fp32[i]);
			clReleaseMemObject(p_ino_hdd4_lyr_data_fp32[i]);
			clReleaseMemObject(p_out_label_s32[i]);
			clReleaseMemObject(p_out_conf_lv_fp32[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseMemObject %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseMemObject %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseKernel(kernel_inp_lyr[i]);
			clReleaseKernel(kernel_hdd1_lyr[i]);
			clReleaseKernel(kernel_hdd2_lyr[i]);
			clReleaseKernel(kernel_hdd3_lyr[i]);
			clReleaseKernel(kernel_out_lyr[i]);
			clReleaseKernel(kernel_red_lyr[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseKernel %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseKernel %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
#if (1 == PROFILING_ENABLE)
			clReleaseEvent(ev_write[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_write[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_write[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
#endif
#if (1 == PROFILING_ENABLE)
			clReleaseEvent(ev_kernel_i[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_kernel_0[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_kernel_0[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseEvent(ev_kernel_1[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_kernel_1[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_kernel_1[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseEvent(ev_kernel_2[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_kernel_2[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_kernel_2[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		clReleaseEvent(ev_kernel_3[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_kernel_3[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_kernel_3[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseEvent(ev_kernel_4[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_kernel_4[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_kernel_4[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseEvent(ev_kernel_r[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_kernel_r[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_kernel_r[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
			clReleaseEvent(ev_read[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseEvent(ev_read[i]) %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseEvent(ev_read[i]) %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif		
#endif
			clReleaseProgram(program[i]);
#if (1 == DEBUGGING_INFO_PRINT)
			printf("clReleaseProgram %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
			clock_gettime(CLOCK_MONOTONIC, &end);
			timespec_subtract(&spent, &end, &start);
			printf("clReleaseProgram %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		}

		free(kernel_inp_lyr);
		free(kernel_hdd1_lyr);
		free(kernel_hdd2_lyr);
		free(kernel_hdd3_lyr);
		free(kernel_out_lyr);
		free(kernel_red_lyr);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("free kernel done\n");
#endif
#if (1 == PROFILING_ENABLE)
		clock_gettime(CLOCK_MONOTONIC, &end);
		timespec_subtract(&spent, &end, &start);
		printf("free kernel time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
		free(p_inp_lyr_data_fp32);
		free(p_inp_hdd0_lyr_wgt_conv_fp32);
		free(p_inp_hdd0_lyr_wgt_bias_fp32);
		free(p_ino_hdd0_lyr_data_fp32);
		free(p_inp_hdd1_lyr_wgt_conv_fp32);
		free(p_inp_hdd1_lyr_wgt_bias_fp32);
		free(p_ino_hdd1_lyr_data_fp32);
		free(p_inp_hdd2_lyr_wgt_conv_fp32);
		free(p_inp_hdd2_lyr_wgt_bias_fp32);
		free(p_ino_hdd2_lyr_data_fp32);
		free(p_inp_hdd3_lyr_wgt_conv_fp32);
		free(p_inp_hdd3_lyr_wgt_bias_fp32);
		free(p_ino_hdd3_lyr_data_fp32);
		free(p_inp_hdd4_lyr_wgt_conv_fp32);
		free(p_inp_hdd4_lyr_wgt_bias_fp32);
		free(p_ino_hdd4_lyr_data_fp32);
		free(p_out_label_s32);
		free(p_out_conf_lv_fp32);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("free mem done\n");
#endif
#if (1 == PROFILING_ENABLE)
		clock_gettime(CLOCK_MONOTONIC, &end);
		timespec_subtract(&spent, &end, &start);
		printf("free mem time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
	}
	else
	{
		CHECK_ERROR(-1);
	}

	/* release common */
	for(i = 0; i < num_devs; i++)
	{
		clReleaseCommandQueue(cmd_queues[i]);
#if (1 == DEBUGGING_INFO_PRINT)
		printf("clReleaseCommandQueue %d done\n", i);
#endif
#if (1 == PROFILING_ENABLE)
		clock_gettime(CLOCK_MONOTONIC, &end);
		timespec_subtract(&spent, &end, &start);
		printf("clReleaseCommandQueue %d time: %ld.%03ld sec\n", i, spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
	}

	clReleaseContext(context);
#if (1 == DEBUGGING_INFO_PRINT)
	printf("clReleaseContext done\n");
#endif
#if (1 == PROFILING_ENABLE)
	clock_gettime(CLOCK_MONOTONIC, &end);
	timespec_subtract(&spent, &end, &start);
	printf("clReleaseContext time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
	
	free(platform);
#if (1 == DEBUGGING_INFO_PRINT)
	printf("free platform done\n");
#endif
#if (1 == PROFILING_ENABLE)
	clock_gettime(CLOCK_MONOTONIC, &end);
	timespec_subtract(&spent, &end, &start);
	printf("free platform time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
	free(devs);
#if (1 == DEBUGGING_INFO_PRINT)
	printf("free devs done\n");
#endif
#if (1 == PROFILING_ENABLE)
	clock_gettime(CLOCK_MONOTONIC, &end);
	timespec_subtract(&spent, &end, &start);
	printf("free devs time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
	free(cmd_queues);
#if (1 == DEBUGGING_INFO_PRINT)
	printf("free cmd_queues done\n");
#endif
#if (1 == PROFILING_ENABLE)
	clock_gettime(CLOCK_MONOTONIC, &end);
	timespec_subtract(&spent, &end, &start);
	printf("free cmd_queues time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
#endif
	free(program);
	
	free(weights);
	free(biases);
}
