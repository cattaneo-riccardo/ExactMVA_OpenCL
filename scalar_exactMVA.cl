__kernel void exactMVA_1(
             __global float* response,
             __global const float* demand,
             __global float* num_jobs,
             __global float* partial_sums)
{
    __private int k=get_local_id(0);

    response[k]=mad(demand[k], num_jobs[k], demand[k]);
    partial_sums[k]=response[k];
}

__kernel void exactMVA_2(
             __global float* partial_sums,
                        const uint i)
{
    __private int k=get_local_id(0);

    if(k < i)
    {
        partial_sums[k] += partial_sums[k + i];
    }
}

__kernel void exactMVA_3(
             __global float* response,
             __global float* num_jobs,
             __global float* partial_sums,

             uint const jobs,
             float const think_time)
{
    __private int k=get_local_id(0);
    __private global_thr;

    global_thr=jobs/(think_time + partial_sums[0]);
    num_jobs[k]=global_thr*response[k];
}
