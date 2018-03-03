__kernel void global_exactMVA(
             __global float* response,
             __global const float* demand,
             __global float* num_jobs,
             __global float* partial_sums,

             uint const segment,
             ulong const tot_jobs,
             float const think_time)
{
    __private uint const lid=get_local_id(0);
    __private uint const min_k=lid*segment;
    __private uint const max_k=min_k+segment-1;
    __private uint const half_group_size = get_local_size(0)/2;

    __private float global_thr=0.0;

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
             __global float* response,
             __global const float* demand,
             __local float* num_jobs,
             __local float* partial_sums,

             uint const segment,
             ulong const tot_jobs,
             float const think_time)
{
    __private uint const lid=get_local_id(0);
    __private uint const min_k=lid*segment;
    __private uint const max_k=min_k+segment-1;
    __private uint const half_group_size = get_local_size(0)/2;

    __private float global_thr=0.0;

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
             __global float* response,
             __global const float* demand,
             __local float* num_jobs,
             __local float* partial_sums,

             uint const segment, /*Not Used*/
             ulong const tot_jobs, /*N*/
             float const think_time) /*Z*/
{
    __private uint k=get_local_id(0);
    __private uint const half_group_size = get_local_size(0)/2;

    __private float global_thr; /*X*/
    __private float local_response; /*Rk*/

    num_jobs[k]=0; /*Nk*/

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
