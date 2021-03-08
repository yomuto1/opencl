#include <stdio.h>
#include <time.h>

#include "sgemm.h"
#include "sgemm_common_def.h"

static float sa_a_f32[SIZE_K * SIZE_M];
static float sa_b_f32[SIZE_K * SIZE_N];
static float sa_alg_c_f32[SIZE_M * SIZE_N];
static float sa_ocl_c_f32[SIZE_M * SIZE_N];

int main(int argc, char ** argv)
{
	int i_s32 = 0;
    clock_t start_point, end_point;

   	// Initialize values for array members.
	for (i_s32 = 0; i_s32 < SIZE_K * SIZE_M; ++i_s32)
    {
		sa_a_f32[i_s32] = (i_s32 * 0.32f) - 1000.f;
	}
	for (i_s32 = 0; i_s32 < SIZE_K * SIZE_N; ++i_s32)
	{
		sa_b_f32[i_s32] = 2500.f - (i_s32 * 0.27f);
	}

    start_point = clock();

    sgemm_alg(sa_a_f32, sa_b_f32, sa_alg_c_f32);

    end_point = clock();

    printf("Exe time alg: %f sec\n", ((float)(end_point - start_point)/CLOCKS_PER_SEC));

    start_point = clock();

	sgemm_ocl(sa_a_f32, sa_b_f32, sa_ocl_c_f32);

    end_point = clock();

    printf("Exe time ocl: %f sec\n", ((float)(end_point - start_point)/CLOCKS_PER_SEC));

	// Test if correct answer
	for (i_s32 = 0; i_s32 < SIZE_M * SIZE_N; ++i_s32)
    {
		if (sa_alg_c_f32[i_s32] != sa_ocl_c_f32[i_s32])
        {
			printf("mismatch: %d  %f %f\n", i_s32, sa_alg_c_f32[i_s32], sa_ocl_c_f32[i_s32]);
			break;
		}
	}
	if (i_s32 == SIZE_M * SIZE_N)
    {
		printf("Everything seems to work fine! \n");
	}

    return 0;
}