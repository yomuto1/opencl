#define MAX_NO_DATA       (1048576)
#define SZ_MAX_CENTROID   (32)

#define SZ_MAX_CU         (32)
#define SZ_MAX_PE         (64)
#define SZ_MAX_DAT_TRANSF (MAX_NO_DATA / (SZ_MAX_CU * SZ_MAX_PE))


__kernel __attribute__((vec_type_hint(float4))) void kmeans_assign(	__global        float4* g_src_cent_arr_f32x2x2,
																	__global        float4* g_src_data_arr_f32x2x2, 
																	__global        int2*   p_res_part_arr_s32x2,
																	__private       int     sz_max_class, 
																	__private       int     sz_max_src_data)
{
	int pos_cur_src = get_global_id(0);
	int pos_local_id = get_local_id(0);

	if(pos_cur_src < sz_max_src_data)
	{
		float  min_dist_x = DBL_MAX;
		float  min_dist_y = DBL_MAX;
		int    min_class_x = 0;
		int    min_class_y = 0;
		float4 diff4, cent4;
		float  dist;
		int class_i;		
		__local float4 l_tmp_cent_buf[SZ_MAX_CENTROID];
        
        float4 src_data_f32x2x2 = 0;
        float4 cur_cent_f32x2x2 = 0;
		
		if(pos_local_id < sz_max_class)
		{
			l_tmp_cent_buf[pos_local_id] = g_src_cent_arr_f32x2x2[pos_local_id];
		}

		/* ----------------------- */
		barrier(CLK_LOCAL_MEM_FENCE);
		/* ----------------------- */
		
		for(class_i = 0 ; class_i < sz_max_class ; class_i++)
		{
            src_data_f32x2x2 = g_src_data_arr_f32x2x2[pos_cur_src];
            cur_cent_f32x2x2 = l_tmp_cent_buf[class_i];
            
			cent4.xyzw = cur_cent_f32x2x2.xyxy;
			diff4 = src_data_f32x2x2 - cent4;
			dist = dot(diff4.lo, diff4.lo);
			
			if(dist < min_dist_x)
			{
				min_class_x = class_i << 1;
				min_dist_x = dist;
			}

			dist = dot(diff4.hi, diff4.hi);

			if(dist < min_dist_y)
			{
				min_class_y = class_i << 1;
				min_dist_y = dist;
			}

			cent4.xyzw = cur_cent_f32x2x2.zwzw;
			diff4 = src_data_f32x2x2 - cent4;
			dist = dot(diff4.lo, diff4.lo);

			if(dist < min_dist_x)
			{
				min_class_x = (class_i << 1) + 1;
				min_dist_x = dist;
			}
			
			dist = dot(diff4.hi, diff4.hi);

			if(dist < min_dist_y)
			{
				min_class_y = (class_i << 1) + 1;
				min_dist_y = dist;
			}
		}
		p_res_part_arr_s32x2[pos_cur_src] = (int2)(min_class_x, min_class_y);
	}
}

__kernel void kmeans_update(	__global        float2*  g_res_cent_arr, 
								__global        int*     g_res_count_arr, 
								__global  const float2*  g_src_data_arr, 
								__global  const int*     g_src_part_arr, 
								                int      no_max_class, 
								                int      sz_src_data, 
												int      SZ_SLOT_IN_BANK)
{
	/* index variables */
	int idx_pos_cent    = 0;
	int idx_pos_data    = 0;
	int pos_out_to_save = 0;

	/* variables releated to built-in function */
	int pos_global      = get_global_id(0);
	int sz_stride       = get_global_size(0);
	int pos_base_global = pos_global * SZ_SLOT_IN_BANK;//get_group_id(0) * get_local_size(0);

	/* private data for src data fetch */
	float2 cur_data = 0;
	int    cur_part = 0;
	/* private buffers for accumulation */
	float2 p_tmp_data_sum [SZ_MAX_CENTROID];
	int    p_tmp_count_sum[SZ_MAX_CENTROID];

	/* init private temp buffer */
	for(idx_pos_cent = 0 ; idx_pos_cent < no_max_class ; idx_pos_cent++)
	{
		p_tmp_data_sum [idx_pos_cent] = 0;
		p_tmp_count_sum[idx_pos_cent] = 0;
	}
	
	/* accumulation size: SZ_SLOT_IN_BANK */
	for(idx_pos_data = pos_base_global ; idx_pos_data < (pos_base_global + SZ_SLOT_IN_BANK) ; idx_pos_data++)
	{
		if(idx_pos_data < sz_src_data)
		{
			cur_data = g_src_data_arr[idx_pos_data];
			cur_part = g_src_part_arr[idx_pos_data];
			
			p_tmp_data_sum [cur_part] += cur_data;
			p_tmp_count_sum[cur_part]++;
		}
	}
	
	/* save with conversion to plannar type from the view point of centroid order */
	for(idx_pos_cent = 0 ; idx_pos_cent < no_max_class ; idx_pos_cent++)
	{
		pos_out_to_save = (idx_pos_cent * sz_stride) + pos_global;

		g_res_cent_arr [pos_out_to_save] = p_tmp_data_sum [idx_pos_cent];
		g_res_count_arr[pos_out_to_save] = p_tmp_count_sum[idx_pos_cent];
	}

	/* divide the sum with number of class for mean point */
	/* ==> move to kernel_reduct with plannar order (from the view point of centroid order) */
}

__kernel void kmeans_reduct(	__global        float2*  g_res_cent_arr, 
								__global  const float2*  g_src_cent_stream, 
								__global  const int*     g_src_count_stream, 
								                int      no_max_class, 
                                                int      sz_src_data, 
                                                int      SZ_SRC_STRIDE, 
												int      SZ_SLOT_IN_BANK)
{
	/* variables releated to built-in function */
	int pos_res_cent  = get_group_id(0);
    int pos_sub_id    = get_local_id(0);
    int sz_local_item = get_local_size(0);

	/* position variables */
    int pos_base_global   = (SZ_SRC_STRIDE   * pos_res_cent);
    int pos_base_local    = (SZ_SLOT_IN_BANK * pos_sub_id);
    int pos_base_src      = pos_base_global + pos_base_local;

    /* index variables */
    int idx_pos_data = 0;
    int idx_step     = 0;

	/* private data for src data fetch */
	__private float2 acc_tmp_data  = 0;
	__private int    acc_tmp_count = 0;

	/* local buffers for accumulation */
	__local float2 l_tmp_data_sum [256];
	__local int    l_tmp_count_sum[256];



    /* accumulation with size SZ_SLOT_IN_BANK */
    for(idx_pos_data = pos_base_src ; idx_pos_data < (pos_base_src + SZ_SLOT_IN_BANK) ; idx_pos_data++)
    {
        if(idx_pos_data < sz_src_data)
        {
            acc_tmp_data  += g_src_cent_stream [idx_pos_data];
            acc_tmp_count += g_src_count_stream[idx_pos_data];
        }
    }

    /* initialize local buffers */
    l_tmp_data_sum [pos_sub_id] = acc_tmp_data;
    l_tmp_count_sum[pos_sub_id] = acc_tmp_count;

	/* ----------------------- */
	barrier(CLK_LOCAL_MEM_FENCE);
	/* ----------------------- */


    /* hand-maded reduction */
    for(idx_step = (sz_local_item / 2) ; idx_step >= 1 ; idx_step = idx_step >> 1)
    {
        if(pos_sub_id < idx_step)
		{
			l_tmp_data_sum [pos_sub_id] += l_tmp_data_sum [pos_sub_id + idx_step];
			l_tmp_count_sum[pos_sub_id] += l_tmp_count_sum[pos_sub_id + idx_step];
		}
		/* ----------------------- */
		barrier(CLK_LOCAL_MEM_FENCE);
		/* ----------------------- */
    }

	if(0 == pos_sub_id)
	{
		g_res_cent_arr[pos_res_cent] = l_tmp_data_sum[0] / l_tmp_count_sum[0];
	}
}
