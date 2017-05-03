#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"

#include <CL/cl.h>


#define RUN_TIME_KERNEL_BUILD   (0) /* 0 ==> bin-run or pre build mode, 1 ==> run-time compile mode */
#define PRE_BUILD_MODE          (0 & (0 == RUN_TIME_KERNEL_BUILD)) /* 0 ==> bin run mode, 1 ==> pre build mode */

#define FILE_NAME_KERNEL_CODE   "kmeans.cl"
#define FILE_NAME_KERNEL_BIN    "kmeans.bin"

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






#define NAME_KERNEL_ASSIGN "kmeans_assign"
#define NAME_KERNEL_UPDATE "kmeans_update"
#define NAME_KERNEL_REDUCT "kmeans_reduct"



#define MAX_PLATFORM          (1)
#define MAX_DEVICE            (1)
#define MAX_NAME_BUFFER       (2048)


#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }


char *get_source_code(const char *file_name, size_t *len) {
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

#define MAX_NO_DATA     (1048576)
#define SZ_MAX_CENTROID (32)

#define SZ_MAX_CU       (32)
#define SZ_MAX_PE       (64)
#define SZ_MAX_PES_ALL  (SZ_MAX_CU * SZ_MAX_PE)

#define MAX_SZ_LOCAL_ASSIGN (256)

#define SZ_NO_TASK_IN_WG (4)
	

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned)
{
	/* ----------------------------------- */
	/* start of local variable declaration */
	/* ----------------------------------- */

	/* iteration index */
	cl_int idx_iter = 0;
	/* size divided by 2 */
	cl_int class_n_div2 = class_n >> 1;
	cl_int data_n_div2  = data_n  >> 1;
	/* error */
	cl_int err = CL_SUCCESS;
	/* get num of platform */
	cl_uint num_platforms = 0;
	/* get num of device */
	cl_uint num_devices = 0;


	/* get platform ID */
	cl_platform_id platform = NULL;
	/* get num of device */
	cl_device_id device = NULL;
	/* context */
	cl_context context = NULL;
	/* cmd queue */
	cl_command_queue queue = NULL;
	/* get kernel source */
	cl_program program = NULL;

#if (RUN_WITH_CL_CODE == RUN_MODE)
	size_t sz_kernel_src = 0;
	char *code_kernel_src = get_source_code(FILE_NAME_KERNEL_CODE, &sz_kernel_src);
#elif (PRE_BUILD_COMPILE == RUN_MODE)
	size_t sz_kernel_src = 0;
	char *code_kernel_src = get_source_code(FILE_NAME_KERNEL_CODE, &sz_kernel_src);
#else /* RUN_WITH_BINARY == RUN_MODE) */
#endif 

	/* kernel object */
	cl_kernel kernel_assign = NULL;
	size_t sz_global_assign = (size_t)((((data_n >> 1) + MAX_SZ_LOCAL_ASSIGN - 1) / MAX_SZ_LOCAL_ASSIGN) * MAX_SZ_LOCAL_ASSIGN);
/*	size_t sz_global_assign = (size_t)((((data_n) + MAX_SZ_LOCAL_ASSIGN - 1) / MAX_SZ_LOCAL_ASSIGN) * MAX_SZ_LOCAL_ASSIGN); */
	size_t sz_local_assign  = (size_t)MAX_SZ_LOCAL_ASSIGN;

	cl_kernel kernel_update       = NULL;
	size_t sz_local_update        = SZ_MAX_PE * SZ_NO_TASK_IN_WG;
	size_t sz_global_update       = SZ_MAX_PES_ALL * SZ_NO_TASK_IN_WG;
	cl_int SZ_SLOT_IN_BANK_UPDATE = ((data_n + (SZ_MAX_PES_ALL * SZ_NO_TASK_IN_WG) - 1) / (SZ_MAX_PES_ALL * SZ_NO_TASK_IN_WG));

	cl_kernel kernel_reduct       = NULL;
	size_t sz_local_reduct        = SZ_MAX_PE;
	size_t sz_global_reduct       = class_n * sz_local_reduct;
    cl_int sz_max_src_reduct      = class_n * SZ_MAX_PES_ALL * SZ_NO_TASK_IN_WG;
    cl_int SZ_STRIDE_REDUCT       = SZ_MAX_PES_ALL * SZ_NO_TASK_IN_WG;
	cl_int SZ_SLOT_IN_BANK_REDUCT = SZ_MAX_CU * SZ_NO_TASK_IN_WG;/* (SZ_MAX_PES_ALL / SZ_MAX_PE); */

	/* memory object */
	cl_mem buf_cen = NULL;
	cl_mem buf_dat = NULL;
	cl_mem buf_par = NULL;
	cl_mem buf_cen_arr = NULL;
	cl_mem buf_cnt_arr = NULL;
	/* --------------------------------- */
	/* end of local variable declaration */
	/* --------------------------------- */

	if(class_n > SZ_MAX_CENTROID)
	{
		printf("class_n(%d) is larger than SZ_MAX_CENTROID(%d). check it!\n", class_n, SZ_MAX_CENTROID);
		exit(1);
	}
	if(data_n > MAX_NO_DATA)
	{
		printf("data_n(%d) is larger than MAX_NO_DATA(%d). check it!\n", data_n, MAX_NO_DATA);
		exit(1);
	}		

	/* ---------------- */
	/* get platform IDs */
	/* ---------------- */
	err = clGetPlatformIDs(1, &platform, &num_platforms);
	CHECK_ERROR(err);

	/* -------------- */
	/* get device IDs */
	/* -------------- */
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
	CHECK_ERROR(err);

	/* -------------- */
	/* create context */
	/* -------------- */
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);

	/* ------------------------- */
	/* create in-order cmd queue */
	/* ------------------------- */
	queue = clCreateCommandQueue(context, device, 0, &err);
	CHECK_ERROR(err);

#if (RUN_WITH_CL_CODE == RUN_MODE)
    

	/* --------------------- */
	/* create program object */
	/* --------------------- */
	program = clCreateProgramWithSource(context, 1, (const char **)&code_kernel_src, &sz_kernel_src, &err);
	CHECK_ERROR(err);
    
	/* ------------------------ */
	/* build kernel source code */
	/* ------------------------ */
	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
#if (1 == CHECK_BUILD_ERROR_LOG)
	//if(CL_BUILD_PROGRAM_FAILURE == err)
	{
		size_t log_size = 0;
		char *log = NULL;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		log = (char *)malloc(log_size + 1);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		log[log_size] = '\0';
		printf(":kernel Compile log:\n%s\n", log);
		free(log);
		log = NULL;
	}
#endif
	CHECK_ERROR(err);

    
#elif (PRE_BUILD_COMPILE == RUN_MODE)



	/* --------------------- */
	/* create program object */
	/* --------------------- */
	program = clCreateProgramWithSource(context, 1, (const char **)&code_kernel_src, &sz_kernel_src, &err);
	CHECK_ERROR(err);
    
	/* ------------------------ */
	/* build kernel source code */
	/* ------------------------ */
	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
#if (1 == CHECK_BUILD_ERROR_LOG)
	//if(CL_BUILD_PROGRAM_FAILURE == err)
	{
		size_t log_size = 0;
		char *log = NULL;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		log = (char *)malloc(log_size + 1);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		log[log_size] = '\0';
		printf(":kernel Compile log:\n%s\n", log);
		free(log);
		log = NULL;
	}
#endif
	CHECK_ERROR(err);

    
	cl_uint nb_devices;
	size_t nbread;
    err = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(size_t), &nb_devices, &nbread);// Return 1 devices  
	CHECK_ERROR(err);
    size_t *np = new size_t[nb_devices];//Create size array   
    err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*nb_devices, np, &nbread);//Load in np the size of my binary  
	CHECK_ERROR(err);
    char** bn = new char* [nb_devices]; //Create the binary array
    for(int i =0; i < nb_devices;i++)  bn[i] = new char[np[i]]; // I know... it's bad... but if i use new char[np[i]], i have a segfault... :/  
    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *)*nb_devices, bn, &nbread); //Load the binary itself  
	CHECK_ERROR(err);
    printf("%s\n", bn[0]); //Print the first binary. But here, I have some curious characters  
    FILE *fp = fopen(FILE_NAME_KERNEL_BIN, "wb");  
    fwrite(bn[0], sizeof(char), np[0], fp); // Save the binary, but my file stay empty  
    fclose(fp);  

	delete np;
	delete bn;

    printf("\n pre-build done !\n change PRE_BUILD_MODE mode as 0\n");
    exit(-1);


#else /* RUN_WITH_BINARY == RUN_MODE) */


	FILE *fp = fopen(FILE_NAME_KERNEL_BIN, "rb");
	size_t binarySize;
    fseek(fp, 0, SEEK_END);
    binarySize = ftell(fp);
    rewind(fp);
	unsigned char *programBinary = (unsigned char *)malloc(binarySize);
    fread(programBinary, 1, binarySize, fp);
	program = clCreateProgramWithBinary(context, 1, &device, &binarySize, (const unsigned char**)&programBinary, NULL, &err);
	CHECK_ERROR(err);
	fclose(fp);

	/* ------------------------ */
	/* build kernel source code */
	/* ------------------------ */
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
#if (1 == CHECK_BUILD_ERROR_LOG)
	//if(CL_BUILD_PROGRAM_FAILURE == err)
	{
		size_t log_size = 0;
		char *log = NULL;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		log = (char *)malloc(log_size + 1);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		log[log_size] = '\0';
		printf(":kernel Compile log:\n%s\n", log);
		free(log);
		log = NULL;
	}
#endif
    CHECK_ERROR(err);



#endif 





	/* -------------------- */
	/* create kernel object */
	/* -------------------- */
	kernel_assign = clCreateKernel(program, NAME_KERNEL_ASSIGN, &err);
	CHECK_ERROR(err);
	kernel_update = clCreateKernel(program, NAME_KERNEL_UPDATE, &err);
	CHECK_ERROR(err);
	kernel_reduct = clCreateKernel(program, NAME_KERNEL_REDUCT, &err);
	CHECK_ERROR(err);

	/* -------------------- */
	/* create buffer object */
	/* -------------------- */
	buf_cen     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Point)  * class_n,                          NULL, &err);
	CHECK_ERROR(err);
	buf_dat     = clCreateBuffer(context, CL_MEM_READ_ONLY,  sizeof(Point)  * data_n,                           NULL, &err);
	CHECK_ERROR(err);
	buf_par     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * data_n,                           NULL, &err);
	CHECK_ERROR(err);
	buf_cnt_arr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * SZ_MAX_CENTROID * SZ_NO_TASK_IN_WG * SZ_MAX_PES_ALL, NULL, &err);
	CHECK_ERROR(err);
	buf_cen_arr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Point)  * SZ_MAX_CENTROID * SZ_NO_TASK_IN_WG * SZ_MAX_PES_ALL, NULL, &err);
	CHECK_ERROR(err);

	/* ------------ */
	/* write buffer */
	/* ------------ */
	err = clEnqueueWriteBuffer(queue, buf_dat, CL_FALSE, 0, sizeof(Point) * data_n,  data,      0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, buf_cen, CL_TRUE,  0, sizeof(Point) * class_n, centroids, 0, NULL, NULL);
	CHECK_ERROR(err);

	/* ----------------------- */
	/* kernel argument setting */
	/* ----------------------- */
	/* kernel_assign */
	err = clSetKernelArg(kernel_assign, 0, sizeof(cl_mem), &buf_cen);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_assign, 1, sizeof(cl_mem), &buf_dat);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_assign, 2, sizeof(cl_mem), &buf_par);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_assign, 3, sizeof(cl_int), &class_n_div2);//class_n);//
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_assign, 4, sizeof(cl_int), &data_n_div2);//data_n);//
	CHECK_ERROR(err);
	
	/* kernel_update */
	err = clSetKernelArg(kernel_update, 0, sizeof(cl_mem), &buf_cen_arr);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_update, 1, sizeof(cl_mem), &buf_cnt_arr);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_update, 2, sizeof(cl_mem), &buf_dat);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_update, 3, sizeof(cl_mem), &buf_par);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_update, 4, sizeof(cl_int), &class_n);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_update, 5, sizeof(cl_int), &data_n);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_update, 6, sizeof(cl_int), &SZ_SLOT_IN_BANK_UPDATE);
	CHECK_ERROR(err);
	
	
	/* kernel_reduct */
	err = clSetKernelArg(kernel_reduct, 0, sizeof(cl_mem), &buf_cen);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_reduct, 1, sizeof(cl_mem), &buf_cen_arr);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_reduct, 2, sizeof(cl_mem), &buf_cnt_arr);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_reduct, 3, sizeof(cl_int), &class_n);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_reduct, 4, sizeof(cl_int), &sz_max_src_reduct);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_reduct, 5, sizeof(cl_int), &SZ_STRIDE_REDUCT);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_reduct, 6, sizeof(cl_int), &SZ_SLOT_IN_BANK_REDUCT);
	CHECK_ERROR(err);


	/* ---------- */
	/* run kernel */
	/* ---------- */
	for(idx_iter = 0 ; idx_iter < iteration_n ; idx_iter++)
	{
        /* ------------------------ */
		/* run kernel -- assignment */
        /* ------------------------ */
		err = clEnqueueNDRangeKernel(queue, kernel_assign, 1, NULL, &sz_global_assign, &sz_local_assign, 0, NULL, NULL);
		CHECK_ERROR(err);

        /* -------------------- */
		/* run kernel -- update */
        /* -------------------- */
		err = clEnqueueNDRangeKernel(queue, kernel_update, 1, NULL, &sz_global_update, &sz_local_update, 0, NULL, NULL);
		CHECK_ERROR(err);
		#if 0
		if(idx_iter == 0)
		{
			int idx_bank;
            int   * cnt_arr = (int   *)malloc(sizeof(int)   * SZ_MAX_CENTROID * SZ_NO_TASK_IN_WG*SZ_MAX_PES_ALL);
            Point * cen_arr = (Point *)malloc(sizeof(Point) * SZ_MAX_CENTROID * SZ_NO_TASK_IN_WG* SZ_MAX_PES_ALL);
			err = clEnqueueReadBuffer(queue, buf_cnt_arr, CL_TRUE,  0, sizeof(int)   * SZ_MAX_CENTROID * SZ_NO_TASK_IN_WG*SZ_MAX_PES_ALL, cnt_arr,   0, NULL, NULL);
			err = clEnqueueReadBuffer(queue, buf_cen_arr, CL_TRUE,  0, sizeof(Point) * SZ_MAX_CENTROID * SZ_NO_TASK_IN_WG*SZ_MAX_PES_ALL, cen_arr,   0, NULL, NULL);

			for(idx_bank = 0 ; idx_bank < 16 ; idx_bank++)
			{
				int sum = 0;
				float fx = 0;
				float fy = 0;
				int xxx;
                int ttt_sum[256] = {0};
                float ttt_fx[256] = {0};
                float ttt_fy[256] = {0};
                int idx_sss = 0;
				for(xxx = 0 ; xxx < SZ_MAX_PES_ALL*SZ_NO_TASK_IN_WG ; xxx++)
				{
                    idx_sss = xxx/32/SZ_NO_TASK_IN_WG;
                    ttt_sum[idx_sss] += cnt_arr[idx_bank  * SZ_MAX_PES_ALL + xxx];
                    ttt_fx[idx_sss]  += cen_arr[idx_bank  * SZ_MAX_PES_ALL + xxx].x;
                    ttt_fy[idx_sss]  += cen_arr[idx_bank  * SZ_MAX_PES_ALL + xxx].y;

					sum += cnt_arr[idx_bank * SZ_MAX_PES_ALL + xxx];
					fx  += cen_arr[idx_bank * SZ_MAX_PES_ALL + xxx].x;
					fy  += cen_arr[idx_bank * SZ_MAX_PES_ALL + xxx].y;
                }
                if(idx_bank == 0)
                {
                    for(xxx = 0 ; xxx < 64 ; xxx++)
                    {
                        printf("   %4d  %4d  %11.3f %11.3f\n", xxx, ttt_sum[xxx], ttt_fx[xxx], ttt_fy[xxx]);
                    }
                    printf("RRR %2d ==> [%6d] [%11.3f %11.3f]\n", idx_bank, sum, fx, fy);
                }
			}
            free(cnt_arr);
            free(cen_arr);
		}
		#endif

        /* ----------------------- */
		/* run kernel -- reduction */
        /* ----------------------- */
		err = clEnqueueNDRangeKernel(queue, kernel_reduct, 1, NULL, &sz_global_reduct, &sz_local_reduct, 0, NULL, NULL);
		CHECK_ERROR(err);
		#if 0
		if(idx_iter == 0)
		{
            int idx_bank;
			err = clEnqueueReadBuffer(queue, buf_cen, CL_TRUE,  0, sizeof(Point)  * class_n, centroids,   0, NULL, NULL);
			printf("\n\nclclcl\n");
			for(idx_bank = 0 ; idx_bank < class_n ; idx_bank++)
			{
				printf("%2d ==> [%11.3f %11.3f]\n", idx_bank, centroids[idx_bank].x, centroids[idx_bank].y);
			}
		}
		#endif
	}		

	/* ----------- */
	/* read buffer */
	/* ----------- */
	err = clEnqueueReadBuffer(queue, buf_par, CL_FALSE, 0, sizeof(cl_int) * data_n,  partitioned, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueReadBuffer(queue, buf_cen, CL_TRUE,  0, sizeof(Point)  * class_n, centroids,   0, NULL, NULL);
	CHECK_ERROR(err);

	/* ------------- */
	/* release stage */
	/* ------------- */
	clReleaseMemObject(buf_cen);
	clReleaseMemObject(buf_dat);
	clReleaseMemObject(buf_par);
	clReleaseMemObject(buf_cen_arr);
	clReleaseMemObject(buf_cnt_arr);
	clReleaseKernel(kernel_assign);
	clReleaseKernel(kernel_update);
	clReleaseKernel(kernel_reduct);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

}
