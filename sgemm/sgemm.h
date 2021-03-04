#define K (1024 * 1)
#define M (1024 * 2)
#define N (1024 * 4)

int sgemm_alg(const float* p_a_f32, const float* p_b_f32, float* p_c_f32);
int sgemm_ocl(const float* p_a_f32, const float* p_b_f32, float* p_c_f32);
