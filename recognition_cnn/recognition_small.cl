/* --------------------------------------- */
/* tuning parameter for kernel performance */
/* select the optimal parameter!!          */
/* --------------------------------------- */
#define REG_SLOT_PER_PE_FOR_SMALL_INP_LYR					(2) /* 1, 2 only, threshold value for size of small input node */
#define REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR					(2) /* 1, 2 only, threshold value for size of medium hidden node */
#define REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR					(5) /* 1, 2, 5 only, threshold value for size of large output node */

#define USE_OPENCL_KERNEL_INTRINSIC							(0) /* run flag to use manual OPENCL intrinsic */

/* --------------------------------------- */
/* tuning parameter for kernel performance */
/* select the optimal parameter!!          */
/* --------------------------------------- */


#if 1
#define sigmoid(x) (native_recip(1 + native_exp(-x)))
#else
#define sigmoid(x) (1 / (1 + exp(-x)))
#endif


/* --------------------------- */
/* fixed definition for kernel */
/* !!!!! - DO NOT CHANGE !!!!! */
/* --------------------------- */
#define SZ_MAX_INPUT_NODE									(784)
#define SZ_MAX_SMALL_NODE									(64)
#define SZ_MAX_DIGIT_NODE									(10)

#define SZ_STEP_FOR_DEFAULT_REDUCTION						(8)

#define MAX_SZ_INP_LYR_COUNT								(12500)

#define SZ_MAX_CU         									(32)
#define SZ_MAX_PE         									(64)


/* ------------------------------------- */
/* auto-calculated definition for kernel */
/* ------------------------------------- */
#define SZ_MAX_PES_PER_GPU									(SZ_MAX_CU * SZ_MAX_PE)


#define SZ_MAX_SLOT_PER_CU_FOR_INP_LYR_COUNT				(MAX_SZ_INP_LYR_COUNT / SZ_MAX_CU)
#define SZ_TOTAL_ELMT_IN_MAIN_LOOP_FOR_INP_LYR_COUNT		(SZ_MAX_SLOT_PER_CU_FOR_INP_LYR_COUNT * SZ_MAX_CU)
#define SZ_MAX_ELMT_IN_REST_LOOP_FOR_INP_LYR_COUNT			(MAX_SZ_INP_LYR_COUNT - SZ_TOTAL_ELMT_IN_MAIN_LOOP_FOR_INP_LYR_COUNT)

#define SZ_MAX_SLOT_PER_CU_FOR_SMALL_OUT_NODE				(SZ_MAX_SMALL_NODE / SZ_MAX_CU)

#define SZ_TOTAL_ELMT_IN_SLOT_FOR_SMALL_INP_CONV_WGT		(SZ_MAX_INPUT_NODE * REG_SLOT_PER_PE_FOR_SMALL_INP_LYR)
#define SZ_TOTAL_ELMT_IN_SLOT_FOR_SMALL_HDN_CONV_WGT		(SZ_MAX_SMALL_NODE * REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR)
#define SZ_TOTAL_ELMT_IN_SLOT_FOR_SMALL_OUT_CONV_WGT		(SZ_MAX_SMALL_NODE * REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR)

#define SZ_MAX_SLOT_PER_PE_FOR_SMALL_INP_CONV_WGT			(SZ_TOTAL_ELMT_IN_SLOT_FOR_SMALL_INP_CONV_WGT / SZ_MAX_PE)
#define SZ_TOTAL_ELMT_IN_MAIN_LOOP_FOR_SMALL_INP_CONV_WGT	(SZ_MAX_SLOT_PER_PE_FOR_SMALL_INP_CONV_WGT * SZ_MAX_PE)
#define SZ_MAX_ELMT_IN_REST_LOOP_FOR_SMALL_INP_CONV_WGT		(SZ_TOTAL_ELMT_IN_SLOT_FOR_SMALL_INP_CONV_WGT - SZ_TOTAL_ELMT_IN_MAIN_LOOP_FOR_SMALL_INP_CONV_WGT)

#define SZ_MAX_SLOT_PER_PE_FOR_SMALL_HDN_CONV_WGT			(SZ_TOTAL_ELMT_IN_SLOT_FOR_SMALL_HDN_CONV_WGT / SZ_MAX_PE)
#define SZ_TOTAL_ELMT_IN_MAIN_LOOP_FOR_SMALL_HDN_CONV_WGT	(SZ_MAX_SLOT_PER_PE_FOR_SMALL_HDN_CONV_WGT * SZ_MAX_PE)
#define SZ_MAX_ELMT_IN_REST_LOOP_FOR_SMALL_HDN_CONV_WGT		(SZ_TOTAL_ELMT_IN_SLOT_FOR_SMALL_HDN_CONV_WGT - SZ_TOTAL_ELMT_IN_MAIN_LOOP_FOR_SMALL_HDN_CONV_WGT)

#define SZ_MAX_SLOT_PER_PE_FOR_SMALL_OUT_CONV_WGT			(SZ_TOTAL_ELMT_IN_SLOT_FOR_SMALL_OUT_CONV_WGT / SZ_MAX_PE)
#define SZ_TOTAL_ELMT_IN_MAIN_LOOP_FOR_SMALL_OUT_CONV_WGT	(SZ_MAX_SLOT_PER_PE_FOR_SMALL_OUT_CONV_WGT * SZ_MAX_PE)
#define SZ_MAX_ELMT_IN_REST_LOOP_FOR_SMALL_OUT_CONV_WGT		(SZ_TOTAL_ELMT_IN_SLOT_FOR_SMALL_OUT_CONV_WGT - SZ_TOTAL_ELMT_IN_MAIN_LOOP_FOR_SMALL_OUT_CONV_WGT)


/* input */
#define SZ_MAX_SLOT_PER_PE_FOR_INP							(SZ_MAX_INPUT_NODE / SZ_MAX_PE)
#define SZ_TOTAL_ELMT_IN_MAIN_LOOP_FOR_INP					(SZ_MAX_SLOT_PER_PE_FOR_INP * SZ_MAX_PE)
#define SZ_MAX_ELMT_IN_REST_LOOP_FOR_INP					(SZ_MAX_INPUT_NODE - SZ_TOTAL_ELMT_IN_MAIN_LOOP_FOR_INP)

/* small */
#define SZ_MAX_SLOT_PER_PE_FOR_SMALL						(SZ_MAX_SMALL_NODE / SZ_MAX_PE)
#define SZ_TOTAL_ELMT_IN_MAIN_LOOP_FOR_SMALL				(SZ_MAX_SLOT_PER_PE_FOR_SMALL * SZ_MAX_PE)

/* reduction */
#define SZ_MAX_SLOT_PER_PE_FOR_REDUCTION					(MAX_SZ_INP_LYR_COUNT / SZ_MAX_PES_PER_GPU) /* for 0 ~ 12287 parallel processing for all CU*PE */
#define SZ_TOTAL_ELMT_IN_MAIN_LOOP_FOR_REDUCTION			(SZ_MAX_SLOT_PER_PE_FOR_REDUCTION * SZ_MAX_PES_PER_GPU) /* index for 12288 */
#define SZ_MAX_ELMT_IN_REST_LOOP_FOR_REDUCTION				(MAX_SZ_INP_LYR_COUNT - SZ_TOTAL_ELMT_IN_MAIN_LOOP_FOR_REDUCTION) /* 12288~12499 => 212: rest loop proc */


/* for all */
#define FP32 float
#define FP32X2 float2
#define FP32X4 float4

#define INT32S int

/* ------------ */
/* SMALL KERNEL */
/* ------------ */
__kernel __attribute__((vec_type_hint(float4))) void kernel_sml_inp_lyr(	__global const FP32* p_inp_lyr_data_fp32, /* [MAX_SZ_INP_LYR_COUNT * SZ_MAX_INPUT_NODE] */
																			__global const FP32* p_inp_wgt_conv_fp32, /* [   SZ_MAX_SMALL_NODE * SZ_MAX_SMALL_NODE] */
																			__global const FP32* p_inp_wgt_bias_fp32, /* [                   1 * SZ_MAX_SMALL_NODE] */
																			__global       FP32* p_out_lyr_data_fp32  /* [MAX_SZ_INP_LYR_COUNT * SZ_MAX_SMALL_NODE] */
																		)
{
	/* for PE, CU indexing */
	__private INT32S pos_cu_id_s32       = get_group_id(0);
	__private INT32S pos_pe_id_in_cu_s32 = get_local_id(0);

	/* for inp count index */
	__global const FP32* p_base_wgt_conv_fp32      = 0;
	__private INT32S idx_base_slot_out_node_s32    = (pos_cu_id_s32 * SZ_MAX_SLOT_PER_CU_FOR_SMALL_OUT_NODE);
	__private INT32S idx_max_slot_out_node_s32     = (idx_base_slot_out_node_s32 + SZ_MAX_SLOT_PER_CU_FOR_SMALL_OUT_NODE);
	__private INT32S idx_slot_out_node_s32         = 0;
	__private INT32S idx_slot_inp_wgt_conv_s32     = 0;
	__private INT32S pos_base_slot_inp_wgt_conv_s32 = 0;
	__local FP32 loc_inp_wgt_conv_fp32[SZ_TOTAL_ELMT_IN_SLOT_FOR_SMALL_INP_CONV_WGT];
	__local FP32 loc_inp_wgt_bias_fp32[REG_SLOT_PER_PE_FOR_SMALL_INP_LYR];

	/* for idx_slot_for_bank_s32 */
	__private INT32S idx_slot_for_bank_s32 = 0;
	
	/* for input layer index */
	__private INT32S idx_elmt_inp_node_s32 = 0;

	/* for input layered data copy from global to local */
	__global const FP32* p_tmp_inp_lyr_data_fp32 = 0;
	__private INT32S idx_slot_inp_data_s32        = 0;
	__private INT32S pos_base_slot_inp_data_s32   = 0;
	__local FP32 loc_inp_lyr_data_fp32[SZ_MAX_INPUT_NODE];

	/* weighted sum */
	__private INT32S idx_red_data_s32 = 0;
	__private FP32 priv_part_wgt_sum_fp32 = 0.f;

	__private INT32S pos_base_tmp_s32 = (pos_pe_id_in_cu_s32 * SZ_STEP_FOR_DEFAULT_REDUCTION);
	__local FP32 loc_tmp_wtd_data_fp32    [REG_SLOT_PER_PE_FOR_SMALL_INP_LYR * SZ_MAX_PE    ];
	__local FP32 loc_tmp_wtd_data_1st_fp32[REG_SLOT_PER_PE_FOR_SMALL_INP_LYR * SZ_MAX_PE / SZ_STEP_FOR_DEFAULT_REDUCTION]; /* temporary result for 1st reduction */

	/* output */
	__private FP32 priv_wgt_sum_fp32 = 0.f;



	/* ---------------------------------------------------------------- */
	/* for each CU, loop for out node slot by REG_SLOT_PER_PE_FOR_SMALL_INP_LYR */
	/* ---------------------------------------------------------------- */
	for(idx_slot_out_node_s32 = idx_base_slot_out_node_s32 ; idx_slot_out_node_s32 < idx_max_slot_out_node_s32 ; idx_slot_out_node_s32 += REG_SLOT_PER_PE_FOR_SMALL_INP_LYR)
	{

		/* -------------------------------------------------------------- */
		/* for each CU, copy coeff for conv and bias from global to local */
		/* -------------------------------------------------------------- */
		pos_base_slot_inp_wgt_conv_s32 = 0;
		p_base_wgt_conv_fp32           = &p_inp_wgt_conv_fp32[(idx_slot_out_node_s32 * SZ_MAX_INPUT_NODE)]; /* different for each CU */

		for(idx_slot_inp_wgt_conv_s32 = 0 ; idx_slot_inp_wgt_conv_s32 < SZ_MAX_SLOT_PER_PE_FOR_SMALL_INP_CONV_WGT ; idx_slot_inp_wgt_conv_s32++){
			loc_inp_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32] = p_base_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32];
			pos_base_slot_inp_wgt_conv_s32 += SZ_MAX_PE;
		}
#if (0 != SZ_MAX_ELMT_IN_REST_LOOP_FOR_SMALL_INP_CONV_WGT)
		if(pos_pe_id_in_cu_s32 < SZ_MAX_ELMT_IN_REST_LOOP_FOR_SMALL_INP_CONV_WGT){
			loc_inp_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32] = p_base_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32];
		}
#endif
		if(pos_pe_id_in_cu_s32 < REG_SLOT_PER_PE_FOR_SMALL_INP_LYR){
			loc_inp_wgt_bias_fp32[pos_pe_id_in_cu_s32] = p_inp_wgt_bias_fp32[idx_slot_out_node_s32 + pos_pe_id_in_cu_s32];
		}


		/* ------------------------------ */
		/* for each CU, loop for inp node */
		/* ------------------------------ */
		for(idx_elmt_inp_node_s32 = 0 ; idx_elmt_inp_node_s32 < MAX_SZ_INP_LYR_COUNT ; idx_elmt_inp_node_s32++)
		{
			/* ---------------------------------------------------- */
			/* for each CU, copy inp node data from global to local */
			/* ---------------------------------------------------- */
			pos_base_slot_inp_data_s32 = 0;
			p_tmp_inp_lyr_data_fp32    = &p_inp_lyr_data_fp32[(idx_elmt_inp_node_s32 * SZ_MAX_INPUT_NODE)];

			for(idx_slot_inp_data_s32 = 0 ; idx_slot_inp_data_s32 < SZ_MAX_SLOT_PER_PE_FOR_INP ; idx_slot_inp_data_s32++){
				loc_inp_lyr_data_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_data_s32] = p_tmp_inp_lyr_data_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_data_s32];
				pos_base_slot_inp_data_s32 += SZ_MAX_PE;
			}
			if(pos_pe_id_in_cu_s32 < SZ_MAX_ELMT_IN_REST_LOOP_FOR_INP){
				loc_inp_lyr_data_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_data_s32] = p_tmp_inp_lyr_data_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_data_s32];
			}
			/* ----------------------- */
			barrier(CLK_LOCAL_MEM_FENCE);
			/* ----------------------- */



			/* ------------ */
			/* weighted sum */
			/* ------------ */
			pos_base_slot_inp_wgt_conv_s32 = 0;
			for(idx_slot_for_bank_s32 = 0 ; idx_slot_for_bank_s32 < REG_SLOT_PER_PE_FOR_SMALL_INP_LYR ; idx_slot_for_bank_s32++)
			{
				priv_part_wgt_sum_fp32     = 0.f;
				pos_base_slot_inp_data_s32 = 0;
				/* main loop */
				for(idx_slot_inp_data_s32 = 0 ; idx_slot_inp_data_s32 < SZ_MAX_SLOT_PER_PE_FOR_INP ; idx_slot_inp_data_s32++)
				{
					priv_part_wgt_sum_fp32 += (loc_inp_lyr_data_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_data_s32] * loc_inp_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32]);
					
					pos_base_slot_inp_data_s32     += SZ_MAX_PE;
					pos_base_slot_inp_wgt_conv_s32 += SZ_MAX_PE;
				}
				/* rest loop */
				if(pos_pe_id_in_cu_s32 < SZ_MAX_ELMT_IN_REST_LOOP_FOR_INP){
					priv_part_wgt_sum_fp32 += (loc_inp_lyr_data_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_data_s32] * loc_inp_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32]);
				}
				pos_base_slot_inp_wgt_conv_s32 += SZ_MAX_ELMT_IN_REST_LOOP_FOR_INP;

				loc_tmp_wtd_data_fp32[pos_pe_id_in_cu_s32 + (idx_slot_for_bank_s32 * SZ_MAX_PE)] = priv_part_wgt_sum_fp32;
			}
			/* ----------------------- */
			barrier(CLK_LOCAL_MEM_FENCE);
			/* ----------------------- */



			/* --------------------------------------------------------------------------- */
			/* reduction: 64(PE) * REG_SLOT_PER_PE_FOR_SMALL_INP_LYR --> REG_SLOT_PER_PE_FOR_SMALL_INP_LYR */
			/* --------------------------------------------------------------------------- */
			/* 1st: 64 * REG_SLOT_PER_PE_FOR_SMALL_INP_LYR --> 8 * REG_SLOT_PER_PE_FOR_SMALL_INP_LYR */
			if(pos_pe_id_in_cu_s32 < (REG_SLOT_PER_PE_FOR_SMALL_INP_LYR * SZ_MAX_PE / SZ_STEP_FOR_DEFAULT_REDUCTION))
			{
				priv_wgt_sum_fp32 = 0;
				for(idx_red_data_s32 = 0 ; idx_red_data_s32 < SZ_STEP_FOR_DEFAULT_REDUCTION ; idx_red_data_s32++){
					priv_wgt_sum_fp32 += loc_tmp_wtd_data_fp32[pos_base_tmp_s32 + idx_red_data_s32];
				}
				
				loc_tmp_wtd_data_1st_fp32[pos_pe_id_in_cu_s32] = priv_wgt_sum_fp32;
			}
			/* ----------------------- */
			barrier(CLK_LOCAL_MEM_FENCE);
			/* ----------------------- */
			/* 2nd: 8 * REG_SLOT_PER_PE_FOR_SMALL_INP_LYR --> REG_SLOT_PER_PE_FOR_SMALL_INP_LYR */
			if(pos_pe_id_in_cu_s32 < REG_SLOT_PER_PE_FOR_SMALL_INP_LYR)
			{
				priv_wgt_sum_fp32 = loc_inp_wgt_bias_fp32[pos_pe_id_in_cu_s32];
				for(idx_red_data_s32 = 0 ; idx_red_data_s32 < SZ_STEP_FOR_DEFAULT_REDUCTION ; idx_red_data_s32++){
					priv_wgt_sum_fp32 += loc_tmp_wtd_data_1st_fp32[pos_base_tmp_s32 + idx_red_data_s32];
				}
				
				p_out_lyr_data_fp32[(idx_elmt_inp_node_s32 * SZ_MAX_SMALL_NODE) + idx_slot_out_node_s32 + pos_pe_id_in_cu_s32] = sigmoid(priv_wgt_sum_fp32);
			}
		}
	}
}

__kernel __attribute__((vec_type_hint(float4))) void kernel_sml_hdn_lyr(	__global const FP32* p_inp_lyr_data_fp32, /* [MAX_SZ_INP_LYR_COUNT * SZ_MAX_SMALL_NODE] */
																			__global const FP32* p_inp_wgt_conv_fp32, /* [   SZ_MAX_SMALL_NODE * SZ_MAX_SMALL_NODE] */
																			__global const FP32* p_inp_wgt_bias_fp32, /* [                   1 * SZ_MAX_SMALL_NODE] */
																			__global       FP32* p_out_lyr_data_fp32  /* [MAX_SZ_INP_LYR_COUNT * SZ_MAX_SMALL_NODE] */
																		)
{
	/* for PE, CU indexing */
	__private INT32S pos_cu_id_s32       = get_group_id(0);
	__private INT32S pos_pe_id_in_cu_s32 = get_local_id(0);

	/* for inp index */
	__global const FP32* p_base_wgt_conv_fp32  = 0;
	__private INT32S idx_base_slot_out_node_s32 = (pos_cu_id_s32 * SZ_MAX_SLOT_PER_CU_FOR_SMALL_OUT_NODE);
	__private INT32S idx_max_slot_out_node_s32  = (idx_base_slot_out_node_s32 + SZ_MAX_SLOT_PER_CU_FOR_SMALL_OUT_NODE);
	__private INT32S idx_slot_out_node_s32      = 0;
	__private INT32S idx_slot_inp_wgt_conv_s32  = 0;
	__private INT32S pos_base_slot_inp_wgt_conv_s32 = 0;
	__local FP32 loc_inp_wgt_conv_fp32[SZ_TOTAL_ELMT_IN_SLOT_FOR_SMALL_HDN_CONV_WGT];
	__local FP32 loc_inp_wgt_bias_fp32[REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR];

	/* for idx_slot_for_bank_s32 */
	__private INT32S idx_slot_for_bank_s32 = 0;
	
	/* for input layer index */
	__private INT32S idx_elmt_inp_node_s32 = 0;

	/* for input layered data copy from global to local */
	__global const FP32* p_tmp_inp_lyr_data_fp32 = 0;
	__private INT32S idx_slot_inp_data_s32        = 0;
	__private INT32S pos_base_slot_inp_data_s32   = 0;
	__local FP32 loc_inp_lyr_data_fp32[SZ_MAX_SMALL_NODE];

	/* weighted sum */
	__private INT32S idx_red_data_s32 = 0;
	__private FP32 priv_part_wgt_sum_fp32 = 0.f;

	__private INT32S pos_base_tmp_s32 = (pos_pe_id_in_cu_s32 * SZ_STEP_FOR_DEFAULT_REDUCTION);
	__local FP32 loc_tmp_wtd_data_fp32    [REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR * SZ_MAX_PE    ];
	__local FP32 loc_tmp_wtd_data_1st_fp32[REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR * SZ_MAX_PE / SZ_STEP_FOR_DEFAULT_REDUCTION]; /* temporary result for 1st reduction */

	/* output */
	__private FP32 priv_wgt_sum_fp32 = 0.f;



	/* ---------------------------------------------------------------- */
	/* for each CU, loop for out node slot by REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR */
	/* ---------------------------------------------------------------- */
	for(idx_slot_out_node_s32 = idx_base_slot_out_node_s32 ; idx_slot_out_node_s32 < idx_max_slot_out_node_s32 ; idx_slot_out_node_s32 += REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR)
	{

		/* -------------------------------------------------------------- */
		/* for each CU, copy coeff for conv and bias from global to local */
		/* -------------------------------------------------------------- */
		pos_base_slot_inp_wgt_conv_s32 = 0;
		p_base_wgt_conv_fp32           = &p_inp_wgt_conv_fp32[(idx_slot_out_node_s32 * SZ_MAX_SMALL_NODE)]; /* different for each CU */

		for(idx_slot_inp_wgt_conv_s32 = 0 ; idx_slot_inp_wgt_conv_s32 < SZ_MAX_SLOT_PER_PE_FOR_SMALL_HDN_CONV_WGT ; idx_slot_inp_wgt_conv_s32++){
			loc_inp_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32] = p_base_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32];
			pos_base_slot_inp_wgt_conv_s32 += SZ_MAX_PE;
		}
#if (0 != SZ_MAX_ELMT_IN_REST_LOOP_FOR_SMALL_HDN_CONV_WGT)
		if(pos_pe_id_in_cu_s32 < SZ_MAX_ELMT_IN_REST_LOOP_FOR_SMALL_HDN_CONV_WGT){
			loc_inp_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32] = p_base_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32];
		}
#endif
		if(pos_pe_id_in_cu_s32 < REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR){
			loc_inp_wgt_bias_fp32[pos_pe_id_in_cu_s32] = p_inp_wgt_bias_fp32[idx_slot_out_node_s32 + pos_pe_id_in_cu_s32];
		}



		/* ------------------------------ */
		/* for each CU, loop for inp node */
		/* ------------------------------ */
		for(idx_elmt_inp_node_s32 = 0 ; idx_elmt_inp_node_s32 < MAX_SZ_INP_LYR_COUNT ; idx_elmt_inp_node_s32++)
		{
			/* ---------------------------------------------------- */
			/* for each CU, copy inp node data from global to local */
			/* ---------------------------------------------------- */
			pos_base_slot_inp_data_s32 = 0;
			p_tmp_inp_lyr_data_fp32    = &p_inp_lyr_data_fp32[(idx_elmt_inp_node_s32 * SZ_MAX_SMALL_NODE)];

			for(idx_slot_inp_data_s32 = 0 ; idx_slot_inp_data_s32 < SZ_MAX_SLOT_PER_PE_FOR_SMALL ; idx_slot_inp_data_s32++){
				loc_inp_lyr_data_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_data_s32] = p_tmp_inp_lyr_data_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_data_s32];
				pos_base_slot_inp_data_s32 += SZ_MAX_PE;
			}
			/* ----------------------- */
			barrier(CLK_LOCAL_MEM_FENCE);
			/* ----------------------- */



			/* ------------ */
			/* weighted sum */
			/* ------------ */
			pos_base_slot_inp_wgt_conv_s32 = 0;
			for(idx_slot_for_bank_s32 = 0 ; idx_slot_for_bank_s32 < REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR ; idx_slot_for_bank_s32++)
			{
				priv_part_wgt_sum_fp32     = 0.f;
				pos_base_slot_inp_data_s32 = 0;
				for(idx_slot_inp_data_s32 = 0 ; idx_slot_inp_data_s32 < SZ_MAX_SLOT_PER_PE_FOR_SMALL ; idx_slot_inp_data_s32++)
				{
					priv_part_wgt_sum_fp32 += (loc_inp_lyr_data_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_data_s32] * loc_inp_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32]);
					
					pos_base_slot_inp_data_s32     += SZ_MAX_PE;
					pos_base_slot_inp_wgt_conv_s32 += SZ_MAX_PE;
				}
				loc_tmp_wtd_data_fp32[pos_pe_id_in_cu_s32 + (idx_slot_for_bank_s32 * SZ_MAX_PE)] = priv_part_wgt_sum_fp32;
			}
			/* ----------------------- */
			barrier(CLK_LOCAL_MEM_FENCE);
			/* ----------------------- */



			/* --------------------------------------------------------------------------- */
			/* reduction: 64(PE) * REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR --> REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR */
			/* --------------------------------------------------------------------------- */
			/* 1st: 64 * REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR --> 8 * REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR */
			if(pos_pe_id_in_cu_s32 < (REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR * SZ_MAX_PE / SZ_STEP_FOR_DEFAULT_REDUCTION))
			{
				priv_wgt_sum_fp32 = 0;
				for(idx_red_data_s32 = 0 ; idx_red_data_s32 < SZ_STEP_FOR_DEFAULT_REDUCTION ; idx_red_data_s32++){
					priv_wgt_sum_fp32 += loc_tmp_wtd_data_fp32[pos_base_tmp_s32 + idx_red_data_s32];
				}
				
				loc_tmp_wtd_data_1st_fp32[pos_pe_id_in_cu_s32] = priv_wgt_sum_fp32;
			}
			priv_wgt_sum_fp32 = loc_inp_wgt_bias_fp32[pos_pe_id_in_cu_s32];

			/* ----------------------- */
			barrier(CLK_LOCAL_MEM_FENCE);
			/* ----------------------- */
			/* 2nd: 8 * REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR --> REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR */
			if(pos_pe_id_in_cu_s32 < REG_SLOT_PER_PE_FOR_SMALL_HDN_LYR)
			{
				for(idx_red_data_s32 = 0 ; idx_red_data_s32 < SZ_STEP_FOR_DEFAULT_REDUCTION ; idx_red_data_s32++){
					priv_wgt_sum_fp32 += loc_tmp_wtd_data_1st_fp32[pos_base_tmp_s32 + idx_red_data_s32];
				}
				
				p_out_lyr_data_fp32[(idx_elmt_inp_node_s32 * SZ_MAX_SMALL_NODE) + idx_slot_out_node_s32 + pos_pe_id_in_cu_s32] = sigmoid(priv_wgt_sum_fp32);
			}
		}
	}
}

__kernel __attribute__((vec_type_hint(float4))) void kernel_sml_out_lyr(	__global const FP32* p_inp_lyr_data_fp32, /* [MAX_SZ_INP_LYR_COUNT * SZ_MAX_SMALL_NODE] */
																			__global const FP32* p_inp_wgt_conv_fp32, /* [   SZ_MAX_DIGIT_NODE * SZ_MAX_SMALL_NODE] */
																			__global const FP32* p_inp_wgt_bias_fp32, /* [                   1 * SZ_MAX_DIGIT_NODE] */
																			__global       FP32* p_out_lyr_data_fp32  /* [MAX_SZ_INP_LYR_COUNT * SZ_MAX_DIGIT_NODE] */
																		)
{
	/* for PE, CU indexing */
	__private INT32S pos_cu_id_s32       = get_group_id(0);
	__private INT32S pos_pe_id_in_cu_s32 = get_local_id(0);
	/* weighted sum */
	__private FP32 priv_part_wgt_sum_fp32 = 0;

	/* for out index */
	__global const FP32* p_base_wgt_conv_fp32 = 0;
	__private INT32S idx_slot_out_node_s32    = 0;

	__local FP32 loc_inp_wgt_conv_fp32[SZ_TOTAL_ELMT_IN_SLOT_FOR_SMALL_OUT_CONV_WGT];
	__local FP32 loc_inp_wgt_bias_fp32[REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR];

	__private INT32S pos_base_slot_inp_wgt_conv_s32 = 0;

	/* for input layer index */
	__global const FP32* p_tmp_inp_lyr_data_fp32 = 0;
	__private INT32S idx_base_slot_inp_lyr_s32 = (pos_cu_id_s32 * SZ_MAX_SLOT_PER_CU_FOR_INP_LYR_COUNT);
	__private INT32S idx_max_slot_inp_lyr_s32  = (idx_base_slot_inp_lyr_s32 + SZ_MAX_SLOT_PER_CU_FOR_INP_LYR_COUNT);
	__private INT32S idx_slot_inp_lyr_s32      = 0;

	/* for input layered data copy from global to local */
	__private INT32S idx_slot_inp_data_s32        = 0;
	__private INT32S pos_base_slot_inp_data_s32   = 0;
	__local FP32 loc_inp_lyr_data_fp32[SZ_MAX_SMALL_NODE];

	__private INT32S idx_slot_inp_wgt_conv_s32  = 0;

	/* for idx_slot_for_bank_s32 */
	__private INT32S idx_slot_for_bank_s32 = 0;

	__private INT32S idx_red_data_s32 = 0;

	__private INT32S pos_base_tmp_s32 = (pos_pe_id_in_cu_s32 * SZ_STEP_FOR_DEFAULT_REDUCTION);
	__local FP32 loc_tmp_wtd_data_fp32    [REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR * SZ_MAX_PE    ];
	__local FP32 loc_tmp_wtd_data_1st_fp32[REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR * SZ_MAX_PE / SZ_STEP_FOR_DEFAULT_REDUCTION]; /* temporary result for 1st reduction */

	/* output */
	__private FP32 priv_wgt_sum_fp32 = 0.f;


	/* ----------------------------------------------------------- */
	/* for each CU, loop for out node slot by SZ_MAX_DIGIT_NODE */
	/* ---------------------------------------------------------------- */
	for(idx_slot_out_node_s32 = 0 ; idx_slot_out_node_s32 < SZ_MAX_DIGIT_NODE ; idx_slot_out_node_s32 += REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR)
	{
		/* -------------------------------------------------------------- */
		/* for each CU, copy coeff for conv and bias from global to local */
		/* -------------------------------------------------------------- */
		pos_base_slot_inp_wgt_conv_s32 = 0;
		p_base_wgt_conv_fp32           = &p_inp_wgt_conv_fp32[(idx_slot_out_node_s32 * SZ_MAX_SMALL_NODE)]; /* different for each CU */

		for(idx_slot_inp_wgt_conv_s32 = 0 ; idx_slot_inp_wgt_conv_s32 < SZ_MAX_SLOT_PER_PE_FOR_SMALL_OUT_CONV_WGT ; idx_slot_inp_wgt_conv_s32++){
			loc_inp_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32] = p_base_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32];
			pos_base_slot_inp_wgt_conv_s32 += SZ_MAX_PE;
		}

#if (0 != SZ_MAX_ELMT_IN_REST_LOOP_FOR_SMALL_OUT_CONV_WGT)
		if(pos_pe_id_in_cu_s32 < SZ_MAX_ELMT_IN_REST_LOOP_FOR_SMALL_OUT_CONV_WGT){
			loc_inp_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32] = p_base_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32];
		}
#endif
		if(pos_pe_id_in_cu_s32 < REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR){
			loc_inp_wgt_bias_fp32[pos_pe_id_in_cu_s32] = p_inp_wgt_bias_fp32[idx_slot_out_node_s32 + pos_pe_id_in_cu_s32];
		}


		/* ------------------------------ */
		/* for each CU, loop for inp node */
		/* ------------------------------ */
		/* main loop */
		for(idx_slot_inp_lyr_s32 = idx_base_slot_inp_lyr_s32 ; idx_slot_inp_lyr_s32 < idx_max_slot_inp_lyr_s32 ; idx_slot_inp_lyr_s32++)
		{
			/* ---------------------------------------------------- */
			/* for each CU, copy inp node data from global to local */
			/* ---------------------------------------------------- */
			p_tmp_inp_lyr_data_fp32 = &p_inp_lyr_data_fp32[(idx_slot_inp_lyr_s32 * SZ_MAX_SMALL_NODE)];

			pos_base_slot_inp_data_s32 = 0;
			for(idx_slot_inp_data_s32 = 0 ; idx_slot_inp_data_s32 < SZ_MAX_SLOT_PER_PE_FOR_SMALL ; idx_slot_inp_data_s32++){
				loc_inp_lyr_data_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_data_s32] = p_tmp_inp_lyr_data_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_data_s32];
				pos_base_slot_inp_data_s32 += SZ_MAX_PE;
			}
			/* ----------------------- */
			barrier(CLK_LOCAL_MEM_FENCE);
			/* ----------------------- */



			/* ------------ */
			/* weighted sum */
			/* ------------ */
			pos_base_slot_inp_wgt_conv_s32 = 0;
			for(idx_slot_for_bank_s32 = 0 ; idx_slot_for_bank_s32 < REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR ; idx_slot_for_bank_s32++)
			{
				priv_part_wgt_sum_fp32     = 0;
				pos_base_slot_inp_data_s32 = 0;
				for(idx_slot_inp_data_s32 = 0 ; idx_slot_inp_data_s32 < SZ_MAX_SLOT_PER_PE_FOR_SMALL ; idx_slot_inp_data_s32++)
				{
					priv_part_wgt_sum_fp32 += (loc_inp_lyr_data_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_data_s32] * loc_inp_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32]);

					pos_base_slot_inp_data_s32     += SZ_MAX_PE;
					pos_base_slot_inp_wgt_conv_s32 += SZ_MAX_PE;
				}
				loc_tmp_wtd_data_fp32[pos_pe_id_in_cu_s32 + (idx_slot_for_bank_s32 * SZ_MAX_PE)] = priv_part_wgt_sum_fp32;
			}
			/* ----------------------- */
			barrier(CLK_LOCAL_MEM_FENCE);
			/* ----------------------- */

			

			/* ------------------------------------------------------------------------------------------- */
			/* reduction: 64(PE) * REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR --> REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR */
			/* ------------------------------------------------------------------------------------------- */
			/* 1st: 64 * REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR --> 8 * REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR */
			if(pos_pe_id_in_cu_s32 < ((REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR * SZ_MAX_PE) / SZ_STEP_FOR_DEFAULT_REDUCTION))
			{
				priv_wgt_sum_fp32 = 0;//loc_tmp_wtd_data_fp32[pos_base_tmp_s32] + loc_tmp_wtd_data_fp32[pos_base_tmp_s32 + 1] + loc_tmp_wtd_data_fp32[pos_base_tmp_s32 + 2] + loc_tmp_wtd_data_fp32[pos_base_tmp_s32 + 3] + loc_tmp_wtd_data_fp32[pos_base_tmp_s32 + 4] + loc_tmp_wtd_data_fp32[pos_base_tmp_s32 + 5] + loc_tmp_wtd_data_fp32[pos_base_tmp_s32 + 6] + loc_tmp_wtd_data_fp32[pos_base_tmp_s32 + 7];
				for(idx_red_data_s32 = 0 ; idx_red_data_s32 < SZ_STEP_FOR_DEFAULT_REDUCTION ; idx_red_data_s32++){
					priv_wgt_sum_fp32 += loc_tmp_wtd_data_fp32[pos_base_tmp_s32 + idx_red_data_s32];
				}
				loc_tmp_wtd_data_1st_fp32[pos_pe_id_in_cu_s32] = priv_wgt_sum_fp32;
			}
			/* ----------------------- */
			barrier(CLK_LOCAL_MEM_FENCE);
			/* ----------------------- */
			/* 2nd: 8 * REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR --> REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR */
			if(pos_pe_id_in_cu_s32 < REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR)
			{
				priv_wgt_sum_fp32 = loc_inp_wgt_bias_fp32[pos_pe_id_in_cu_s32];// + loc_tmp_wtd_data_1st_fp32[pos_base_tmp_s32] + loc_tmp_wtd_data_1st_fp32[pos_base_tmp_s32+1] + loc_tmp_wtd_data_1st_fp32[pos_base_tmp_s32 + 2] + loc_tmp_wtd_data_1st_fp32[pos_base_tmp_s32 + 3] + loc_tmp_wtd_data_1st_fp32[pos_base_tmp_s32 + 4] + loc_tmp_wtd_data_1st_fp32[pos_base_tmp_s32 + 5] + loc_tmp_wtd_data_1st_fp32[pos_base_tmp_s32 + 6] + loc_tmp_wtd_data_1st_fp32[pos_base_tmp_s32 + 7];

				for(idx_red_data_s32 = 0 ; idx_red_data_s32 < SZ_STEP_FOR_DEFAULT_REDUCTION ; idx_red_data_s32++){
					priv_wgt_sum_fp32 += loc_tmp_wtd_data_1st_fp32[pos_base_tmp_s32 + idx_red_data_s32];
				}
				p_out_lyr_data_fp32[(idx_slot_inp_lyr_s32 * SZ_MAX_DIGIT_NODE) + idx_slot_out_node_s32 + pos_pe_id_in_cu_s32] = sigmoid(priv_wgt_sum_fp32);
			}
		}
		/* rest loop */
		if(pos_cu_id_s32 < SZ_MAX_ELMT_IN_REST_LOOP_FOR_INP_LYR_COUNT)
		{
			idx_slot_inp_lyr_s32 = pos_cu_id_s32 + SZ_TOTAL_ELMT_IN_MAIN_LOOP_FOR_INP_LYR_COUNT;

			/* ---------------------------------------------------- */
			/* for each CU, copy inp node data from global to local */
			/* ---------------------------------------------------- */
			p_tmp_inp_lyr_data_fp32 = &p_inp_lyr_data_fp32[(idx_slot_inp_lyr_s32 * SZ_MAX_SMALL_NODE)];

			pos_base_slot_inp_data_s32 = 0;

			for(idx_slot_inp_data_s32 = 0 ; idx_slot_inp_data_s32 < SZ_MAX_SLOT_PER_PE_FOR_SMALL ; idx_slot_inp_data_s32++){
				loc_inp_lyr_data_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_data_s32] = p_tmp_inp_lyr_data_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_data_s32];
				pos_base_slot_inp_data_s32 += SZ_MAX_PE;
			}


			/* ------------ */
			/* weighted sum */
			/* ------------ */
			pos_base_slot_inp_wgt_conv_s32 = 0;
			for(idx_slot_for_bank_s32 = 0 ; idx_slot_for_bank_s32 < REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR ; idx_slot_for_bank_s32++)
			{
				priv_part_wgt_sum_fp32     = 0.f;
				pos_base_slot_inp_data_s32 = 0;
				for(idx_slot_inp_data_s32 = 0 ; idx_slot_inp_data_s32 < SZ_MAX_SLOT_PER_PE_FOR_SMALL ; idx_slot_inp_data_s32++)
				{
					priv_part_wgt_sum_fp32 += (loc_inp_lyr_data_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_data_s32] * loc_inp_wgt_conv_fp32[pos_pe_id_in_cu_s32 + pos_base_slot_inp_wgt_conv_s32]);
					
					pos_base_slot_inp_data_s32     += SZ_MAX_PE;
					pos_base_slot_inp_wgt_conv_s32 += SZ_MAX_PE;
				}
				loc_tmp_wtd_data_fp32[pos_pe_id_in_cu_s32 + (idx_slot_for_bank_s32 * SZ_MAX_PE)] = priv_part_wgt_sum_fp32;
			}
			/* ----------------------- */
			barrier(CLK_LOCAL_MEM_FENCE);
			/* ----------------------- */



			/* ------------------------------------------------------------------------------------------- */
			/* reduction: 64(PE) * REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR --> REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR */
			/* ------------------------------------------------------------------------------------------- */
			/* 1st: 64 * REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR --> 8 * REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR */
			if(pos_pe_id_in_cu_s32 < ((REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR * SZ_MAX_PE) / SZ_STEP_FOR_DEFAULT_REDUCTION))
			{
				priv_wgt_sum_fp32 = 0;
				for(idx_red_data_s32 = 0 ; idx_red_data_s32 < SZ_STEP_FOR_DEFAULT_REDUCTION ; idx_red_data_s32++){
					priv_wgt_sum_fp32 += loc_tmp_wtd_data_fp32[pos_base_tmp_s32 + idx_red_data_s32];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			if(pos_pe_id_in_cu_s32 < ((REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR * SZ_MAX_PE) / SZ_STEP_FOR_DEFAULT_REDUCTION))
			{

				loc_tmp_wtd_data_1st_fp32[pos_pe_id_in_cu_s32] = priv_wgt_sum_fp32;
			}
			priv_wgt_sum_fp32 = loc_inp_wgt_bias_fp32[pos_pe_id_in_cu_s32];
			/* ----------------------- */
			barrier(CLK_LOCAL_MEM_FENCE);
			/* ----------------------- */
			/* 2nd: 8 * REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR --> REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR */
			if(pos_pe_id_in_cu_s32 < REG_SLOT_PER_PE_FOR_SMALL_OUT_LYR)
			{
				for(idx_red_data_s32 = 0 ; idx_red_data_s32 < SZ_STEP_FOR_DEFAULT_REDUCTION ; idx_red_data_s32++){
					priv_wgt_sum_fp32 += loc_tmp_wtd_data_1st_fp32[pos_base_tmp_s32 + idx_red_data_s32];
				}
				
				p_out_lyr_data_fp32[(idx_slot_inp_lyr_s32 * SZ_MAX_DIGIT_NODE) + idx_slot_out_node_s32 + pos_pe_id_in_cu_s32] = sigmoid(priv_wgt_sum_fp32);
			}
		}
	}
}

/* ------------- */
/* COMMON KERNEL */
/* ------------- */
__kernel __attribute__((vec_type_hint(float4))) void kernel_reduction_lyr(	__global const FP32*   p_inp_hdd_lyr_data_fp32, /* [MAX_SZ_INP_LYR_COUNT * SZ_MAX_DIGIT_NODE   ] */
																			__global       INT32S* p_out_label_data_s32,    /* [                       MAX_SZ_INP_LYR_COUNT] */
																			__global       FP32*   p_out_conf_lv_fp32       /* [                       MAX_SZ_INP_LYR_COUNT] */
																		)
{
	/* for PE, CU indexing */
	__private INT32S pos_pe_in_all_s32 = get_global_id(0);

	/* for blk loop */
	__private INT32S pos_base_s32     = SZ_MAX_SLOT_PER_PE_FOR_REDUCTION * pos_pe_in_all_s32;
	__private INT32S idx_max_blk_s32  = SZ_MAX_SLOT_PER_PE_FOR_REDUCTION + pos_base_s32;
	__private INT32S idx_blk_s32      = pos_base_s32;
	__private INT32S idx_rest_blk_s32 = (SZ_TOTAL_ELMT_IN_MAIN_LOOP_FOR_REDUCTION + pos_pe_in_all_s32);

	__private INT32S max_lable_s32  = 0;
	__private FP32 max_conf_lv_fp32 = 0.f;
	__private FP32 tmp_conf_lvs_fp32[SZ_MAX_DIGIT_NODE];


	/* base position for inp confence array */
	__private INT32S pos_base_inp_data_s32 = (pos_base_s32 * SZ_MAX_DIGIT_NODE);

	/* main loop processing for all PE*CU */
	for(idx_blk_s32 = pos_base_s32 ; idx_blk_s32 < idx_max_blk_s32 ; idx_blk_s32++)
	{
		/* copy from global to private memory */
		tmp_conf_lvs_fp32[0] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32    ];
		tmp_conf_lvs_fp32[1] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 1];
		tmp_conf_lvs_fp32[2] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 2];
		tmp_conf_lvs_fp32[3] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 3];
		tmp_conf_lvs_fp32[4] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 4];
		tmp_conf_lvs_fp32[5] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 5];
		tmp_conf_lvs_fp32[6] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 6];
		tmp_conf_lvs_fp32[7] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 7];
		tmp_conf_lvs_fp32[8] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 8];
		tmp_conf_lvs_fp32[9] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 9];
		
		/* fully loop unrolling from 0 to 9 by hand */
		max_lable_s32    = 0;								/* 0 */
		max_conf_lv_fp32 = tmp_conf_lvs_fp32[0];
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[1]){		/* 1 */
			max_lable_s32 = 1;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[1];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[2]){		/* 2 */
			max_lable_s32 = 2;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[2];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[3]){		/* 3 */
			max_lable_s32 = 3;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[3];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[4]){		/* 4 */
			max_lable_s32 = 4;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[4];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[5]){		/* 5 */
			max_lable_s32 = 5;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[5];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[6]){		/* 6 */
			max_lable_s32 = 6;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[6];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[7]){		/* 7 */
			max_lable_s32 = 7;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[7];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[8]){		/* 8 */
			max_lable_s32 = 8;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[8];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[9]){		/* 9 */
			max_lable_s32 = 9;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[9];
		}
		/* save output */
		p_out_label_data_s32[idx_blk_s32] = max_lable_s32;
		p_out_conf_lv_fp32  [idx_blk_s32] = max_conf_lv_fp32;

		/* fetch base index update */
		pos_base_inp_data_s32 += SZ_MAX_DIGIT_NODE;
	}

	/* rest loop processing for avaliable PE(some CU) only */
	if(pos_pe_in_all_s32 < SZ_MAX_ELMT_IN_REST_LOOP_FOR_REDUCTION)
	{
		/* base position for inp confence array */
		pos_base_inp_data_s32 = (idx_rest_blk_s32 * SZ_MAX_DIGIT_NODE);

		/* copy from global to private memory */
		tmp_conf_lvs_fp32[0] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32    ];
		tmp_conf_lvs_fp32[1] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 1];
		tmp_conf_lvs_fp32[2] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 2];
		tmp_conf_lvs_fp32[3] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 3];
		tmp_conf_lvs_fp32[4] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 4];
		tmp_conf_lvs_fp32[5] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 5];
		tmp_conf_lvs_fp32[6] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 6];
		tmp_conf_lvs_fp32[7] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 7];
		tmp_conf_lvs_fp32[8] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 8];
		tmp_conf_lvs_fp32[9] = p_inp_hdd_lyr_data_fp32[pos_base_inp_data_s32 + 9];
		
		/* fully loop unrolling from 0 to 9 by hand */
		max_lable_s32    = 0;								/* 0 */
		max_conf_lv_fp32 = tmp_conf_lvs_fp32[0];
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[1]){		/* 1 */
			max_lable_s32 = 1;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[1];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[2]){		/* 2 */
			max_lable_s32 = 2;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[2];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[3]){		/* 3 */
			max_lable_s32 = 3;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[3];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[4]){		/* 4 */
			max_lable_s32 = 4;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[4];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[5]){		/* 5 */
			max_lable_s32 = 5;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[5];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[6]){		/* 6 */
			max_lable_s32 = 6;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[6];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[7]){		/* 7 */
			max_lable_s32 = 7;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[7];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[8]){		/* 8 */
			max_lable_s32 = 8;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[8];
		}
		if(max_conf_lv_fp32 < tmp_conf_lvs_fp32[9]){		/* 9 */
			max_lable_s32 = 9;
			max_conf_lv_fp32 = tmp_conf_lvs_fp32[9];
		}
		/* save output */
		p_out_label_data_s32[idx_rest_blk_s32] = max_lable_s32;
		p_out_conf_lv_fp32  [idx_rest_blk_s32] = max_conf_lv_fp32;
	}
}
