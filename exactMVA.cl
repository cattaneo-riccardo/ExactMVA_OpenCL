#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double FLOAT_TYPE;
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
typedef double FLOAT_TYPE;
#else
typedef float FLOAT_TYPE;
#endif

__kernel void global_exactMVA(
             __global FLOAT_TYPE* response,
             __global const FLOAT_TYPE* demand,
             __global FLOAT_TYPE* num_jobs,
             __global FLOAT_TYPE* partial_sums,

             uint const segment,
             ulong const tot_jobs,
             FLOAT_TYPE const think_time)
{
    __private uint const lid=get_local_id(0);
    __private uint const min_k=lid*segment;
    __private uint const max_k=min_k+segment-1;
    __private uint const half_group_size = get_local_size(0)/2;

    __private FLOAT_TYPE global_thr=0.0;

    for (ulong jobs=1; jobs<=tot_jobs; jobs++)
    {

        partial_sums[lid]=0;
        for (uint k=min_k; k<=max_k; k++){
            num_jobs[k]=global_thr*response[k];
            response[k]=mad(demand[k], num_jobs[k], demand[k]);
            partial_sums[lid]+=response[k];
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        for(uint i = half_group_size; i>0; i >>= 1)
        {
            if(lid < i)
            {
                partial_sums[lid] += partial_sums[lid + i];
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
        }

        global_thr=jobs/(think_time + partial_sums[0]);
    }
}

__kernel void local_exactMVA(
             __global FLOAT_TYPE* response,
             __global const FLOAT_TYPE* demand,
             __local FLOAT_TYPE* num_jobs,
             __local FLOAT_TYPE* partial_sums,

             uint const segment,
             ulong const tot_jobs,
             FLOAT_TYPE const think_time)
{
    __private uint const lid=get_local_id(0);
    __private uint const min_k=lid*segment;
    __private uint const max_k=min_k+segment-1;
    __private uint const half_group_size = get_local_size(0)/2;

    __private FLOAT_TYPE global_thr=0.0;

    for (ulong jobs=1; jobs<=tot_jobs; jobs++)
    {

        partial_sums[lid]=0;
        for (uint k=min_k; k<=max_k; k++){
            num_jobs[k]=global_thr*response[k];
            response[k]=mad(demand[k], num_jobs[k], demand[k]);
            partial_sums[lid]+=response[k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for(uint i = half_group_size; i>0; i >>= 1)
        {
            if(lid < i)
            {
                partial_sums[lid] += partial_sums[lid + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        global_thr=jobs/(think_time + partial_sums[0]);
    }

}

__kernel void single_exactMVA(
             __global FLOAT_TYPE* response,
             __global const FLOAT_TYPE* demand,
             __local FLOAT_TYPE* num_jobs,
             __local FLOAT_TYPE* partial_sums,

             uint const segment, /*Not Used*/
             ulong const tot_jobs, /*N*/
             FLOAT_TYPE const think_time) /*Z*/
{
    __private uint k=get_local_id(0);
    __private uint const half_group_size = get_local_size(0)/2;

    __private FLOAT_TYPE global_thr; /*X*/
    __private FLOAT_TYPE local_response; /*Rk*/

    num_jobs[k]=0; /*Nk*/

    /*if (k==0)
    {
        #if defined(cl_khr_fp64)  // Khronos extension available?
        printf("Using Khronos Double\n");
        #elif defined(cl_amd_fp64)  // AMD extension available?
        printf("Using AMD Double\n");
        #else
        printf("Using single-precision Float\n");
        #endif
    }*/

    for (ulong jobs=1; jobs<=tot_jobs; jobs++)
    {

        partial_sums[k]=local_response=mad(demand[k], num_jobs[k], demand[k]);
        barrier(CLK_LOCAL_MEM_FENCE);

        for(uint i = half_group_size; i>0; i >>= 1)
        {
            if(k < i)
            {
                partial_sums[k] += partial_sums[k + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        global_thr=jobs/(think_time + partial_sums[0]);
        num_jobs[k]=global_thr*local_response;
    }

    response[k]=local_response;
}
