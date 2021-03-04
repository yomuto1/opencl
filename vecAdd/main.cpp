#include <stdio.h>
#include <time.h>

#include "vecAdd.h"

static float sa_a_f32[SIZE];
static float sa_b_f32[SIZE];
static float sa_alg_c_f32[SIZE];
static float sa_ocl_c_f32[SIZE];

int main(int argc, char ** argv)
{
	int i_s32 = 0;
    clock_t start_point, end_point;

   	// Initialize values for array members.
	for (i_s32 = 0; i_s32 < SIZE; ++i_s32)
    {
		sa_a_f32[i_s32] = i_s32 + 1.f;
		sa_b_f32[i_s32] = (i_s32 + 1.f) * 2.f;
	}

    start_point = clock();

    vecAdd_alg(sa_a_f32, sa_b_f32, sa_alg_c_f32, SIZE);

    end_point = clock();

    printf("Exe time alg: %f sec\n", ((float)(end_point - start_point)/CLOCKS_PER_SEC));

    start_point = clock();

    vecAdd_ocl(sa_a_f32, sa_b_f32, sa_ocl_c_f32, SIZE);

    end_point = clock();

    printf("Exe time ocl: %f sec\n", ((float)(end_point - start_point)/CLOCKS_PER_SEC));

	// Test if correct answer
	for (i_s32 = 0; i_s32 < SIZE; ++i_s32)
    {
		if (sa_alg_c_f32[i_s32] != sa_ocl_c_f32[i_s32])
        {
			printf("mismatch: %d  %f %f\n", i_s32, sa_alg_c_f32[i_s32], sa_ocl_c_f32[i_s32]);
			break;
		}
	}
	if (i_s32 == SIZE)
    {
		printf("Everything seems to work fine! \n");
	}

    return 0;
}