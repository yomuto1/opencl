#define SIZE (1920 * 1080 * 20)

int vecAdd_alg(const float* p_a_f32, const float* p_b_f32, float* p_c_f32, const int size_s32);
int vecAdd_ocl(const float* p_a_f32, const float* p_b_f32, float* p_c_f32, const int size_s32);
