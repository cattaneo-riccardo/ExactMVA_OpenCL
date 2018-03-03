__kernel void exactMVA(
             __global float* response,
             __global const float* demand,
             __local float* num_jobs,
             __local float* partial_sums,

             uint const segment,
             uint const tot_jobs,
             float const think_time)
{
    __private uint const lid=get_local_id(0);
    __private uint const min_k=lid*segment;
    __private uint const max_k=min_k+segment-1;
    __private uint const half_group_size = get_local_size(0)/2;

    __private float global_thr=0.0;

    for (uint jobs=1; jobs<=tot_jobs; jobs++)
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
