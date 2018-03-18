#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <math.h>
#include <getopt.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifdef DOUBLE
typedef cl_double FLOAT_TYPE;
#else
typedef cl_float FLOAT_TYPE;
#endif

#define DEVICE 0
#define PLATFORM 0
#define MAX_PLATFORMS 2

#define FILENAME_CL "exactMVA.cl"
#define FILENAME_RESIDENCE "./residences.txt"

#define FUNCTION_NAME_SINGLE "single_exactMVA"
#define FUNCTION_NAME_LOCAL "local_exactMVA"
#define FUNCTION_NAME_GLOBAL "global_exactMVA"

#define NUM_STATIONS_DEFAULT 512
#define NUM_JOBS_DEFAULT 6e4
#define THINK_TIME_DEFAULT 0

#define ZERO_APPROX 1e-3
#define MAX_SOURCE_SIZE 5120

#define VERBOSE

cl_uint segment=1;
bool useLocalMemory=false;
int max_workgroup_size;
cl_ulong local_memory_size;

char str_temp[1024];
cl_platform_id platform_id[MAX_PLATFORMS];
cl_device_id device_id;
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_device_fp_config double_support;
cl_context clContext;
cl_kernel clKernel;
cl_command_queue clCommandQue;
cl_program clProgram;

cl_mem response_mem_obj;
cl_mem demand_mem_obj;
cl_mem num_jobs_obj;
cl_mem partial_sum_obj;

FILE *fp;
char *source_str;
size_t source_size;
//Return iterations that where needed to compute the results*/

void exactMVA(std::vector<FLOAT_TYPE> &response, const std::vector<FLOAT_TYPE> &demand, cl_uint num_stations, cl_uint tot_jobs, FLOAT_TYPE think_time=0)
{
    //Initialize number of jobs in each station to zero
    std::vector<FLOAT_TYPE> num_jobs(num_stations, 0.0);
    FLOAT_TYPE thr=0.0;

    //Main cycle of Exact MVA Algorithm
    for (cl_uint jobs=1; jobs<=tot_jobs; jobs++)
    {
        FLOAT_TYPE tot_resp=0.0;;
        for (cl_uint k=0; k<num_stations; k++)
        {
            num_jobs[k]=thr*response[k];
            response[k]=demand[k]*(1+num_jobs[k]); //Num jobs contains the value of the previous iteration
            tot_resp+=response[k];
        }

        thr=((FLOAT_TYPE)jobs)/(think_time + tot_resp);
    }
}

void read_cl_file()
{
  // Load the kernel source code into the array source_str
    fp = fopen(FILENAME_CL, "r");

  if (!fp) {
    fprintf(stdout, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose( fp );
}

void cl_initialization(cl_uint demands_size)
{
    errcode = clGetPlatformIDs(MAX_PLATFORMS, platform_id, NULL);
    if(errcode != CL_SUCCESS) printf("Error getting platform IDs\n");

    errcode = clGetDeviceIDs(platform_id[PLATFORM], CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
    if(errcode != CL_SUCCESS) printf("Error getting device IDs\n");

    errcode = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
    #ifdef VERBOSE
    if(errcode == CL_SUCCESS) printf("Device used: %s\n",str_temp);
    #endif // VERBOSE
    if (errcode!=CL_SUCCESS) printf("Error getting device name\n");

    // Create an OpenCL context
    clContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
    if(errcode != CL_SUCCESS) printf("Error in creating context\n");

    //Create a command-queue
    clCommandQue = clCreateCommandQueue(clContext, device_id, 0, &errcode);
    if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");

    //Verifying which kernel performs better
    errcode= clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_memory_size, 0);
    if(errcode != CL_SUCCESS) printf("Error in getting Maximum Local Memory Size\n");

    //Deciding parameters for kernel choice
    FLOAT_TYPE required_local_size=demands_size*2*sizeof(FLOAT_TYPE);
    useLocalMemory=(required_local_size<=local_memory_size);

}

void cl_load_prog(cl_uint demands_length)
{
    // Create a program from the kernel source
    clProgram = clCreateProgramWithSource(clContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);
    if(errcode != CL_SUCCESS) printf("Error in creating program\n");

    // Build the program
    errcode = clBuildProgram(clProgram, 1, &device_id, "", NULL, NULL);
    if (errcode != CL_SUCCESS) printf("\nError %d: Failed to build program executable\n",errcode);

    if (useLocalMemory)
    {
        clKernel = clCreateKernel(clProgram, FUNCTION_NAME_SINGLE, &errcode);
        //Computing Max WorkGroup Size
        errcode = clGetKernelWorkGroupInfo(clKernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, NULL);
        if(errcode != CL_SUCCESS) printf("Error getting MAX_WORK_GROUP_SIZE\n");
        segment=ceil(((FLOAT_TYPE)demands_length)/max_workgroup_size);

        if (segment>1)
        {
            errcode = clReleaseKernel(clKernel);
            if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
            clKernel = clCreateKernel(clProgram, FUNCTION_NAME_LOCAL, &errcode);

            //Computing Max WorkGroup Size
            errcode = clGetKernelWorkGroupInfo(clKernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, NULL);
            if(errcode != CL_SUCCESS) printf("Error getting MAX_WORK_GROUP_SIZE\n");
            segment=ceil(((FLOAT_TYPE)demands_length)/max_workgroup_size);
        }
    }
    else {
        clKernel = clCreateKernel(clProgram, FUNCTION_NAME_GLOBAL, &errcode);

        errcode = clGetKernelWorkGroupInfo(clKernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, NULL);
        if(errcode != CL_SUCCESS) printf("Error getting MAX_WORK_GROUP_SIZE\n");
        segment=ceil(((FLOAT_TYPE)demands_length)/max_workgroup_size);
    }
    if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");

    #ifdef VERBOSE
    if (segment>1 && useLocalMemory)
        std::cout<<"Called "<<FUNCTION_NAME_LOCAL<<std::endl;
    else if (segment>1)
        std::cout<<"Called "<<FUNCTION_NAME_GLOBAL<<std::endl;
    else
        std::cout<<"Called "<<FUNCTION_NAME_SINGLE<<std::endl;
    #endif // VERBOSE
}

void cl_mem_init(cl_uint demands_size)
{
    response_mem_obj = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(FLOAT_TYPE)*demands_size, NULL, &errcode);
    if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

    demand_mem_obj = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(FLOAT_TYPE)*demands_size, NULL, &errcode);
    if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

    if (!useLocalMemory){
        num_jobs_obj = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(FLOAT_TYPE)*demands_size, NULL, &errcode);
        partial_sum_obj = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(FLOAT_TYPE)*(demands_size/segment), NULL, &errcode);
        if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");
    }
}

void cl_clean_up()
{
  // Clean up
    errcode = clReleaseKernel(clKernel);
    errcode |= clReleaseProgram(clProgram);
    errcode |= clReleaseMemObject(response_mem_obj);
    errcode |= clReleaseMemObject(demand_mem_obj);
    if (!useLocalMemory){
        errcode |= clReleaseMemObject(num_jobs_obj);
        errcode |= clReleaseMemObject(partial_sum_obj);
    }

    errcode |= clReleaseCommandQueue(clCommandQue);
    errcode |= clReleaseContext(clContext);
    free(source_str);
    if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}

void cl_launch_kernel(std::vector<FLOAT_TYPE> &residences, const std::vector<FLOAT_TYPE> &demands, cl_uint num_stations, cl_uint tot_jobs, FLOAT_TYPE think_time=0)
{
    cl_uint demands_length=demands.size();
    //Computing WorkGroupSize and GlobalWorkSize
    size_t localWorkSize, globalWorkSize;
    localWorkSize = demands_length/segment;
    globalWorkSize = demands_length/segment;

    #ifdef VERBOSE
    std::cout<<"Number of stations: "<<num_stations<<std::endl;
    std::cout<<"Demand Length: "<<demands_length<<std::endl;
    std::cout<<"Segment size: "<<segment<<std::endl;
    std::cout<<"Work Group Size:  "<<localWorkSize<<std::endl;
    std::cout.flush();
    #endif // VERBOSE

    //Writing Demands to device
    errcode = clEnqueueWriteBuffer(clCommandQue, demand_mem_obj, CL_FALSE, 0, demands_length*sizeof(FLOAT_TYPE), &demands[0], 0, NULL, NULL);
    if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");

    // Set the arguments of the kernel
    errcode =  clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&response_mem_obj);
    errcode |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&demand_mem_obj);
    if (useLocalMemory)
    {
        errcode |= clSetKernelArg(clKernel, 2, sizeof(FLOAT_TYPE)*demands_length, NULL); //Local memory space
        errcode |= clSetKernelArg(clKernel, 3, sizeof(FLOAT_TYPE)*(demands_length/segment), NULL); //Local memory space
    }
    else
    {
        errcode |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&num_jobs_obj);
        errcode |= clSetKernelArg(clKernel, 3, sizeof(cl_mem), (void *)&partial_sum_obj);
    }
    errcode |= clSetKernelArg(clKernel, 4, sizeof(cl_uint), &segment);
    errcode |= clSetKernelArg(clKernel, 5, sizeof(cl_uint), &tot_jobs);
    errcode |= clSetKernelArg(clKernel, 6, sizeof(FLOAT_TYPE), &think_time);
    //Launching kernel execution
    if(errcode != CL_SUCCESS) printf("Error in setting arguments: %d\n", errcode);
    errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    //Reading back results
    errcode |= clEnqueueReadBuffer(clCommandQue, response_mem_obj, CL_TRUE, 0, sizeof(FLOAT_TYPE)*num_stations, &residences[0], 0, NULL, NULL);
    if(errcode != CL_SUCCESS)
    {
        printf("Error in reading GPU mem, ID: %d\n", errcode);
        cl_clean_up();
        exit(-1);
    }
}

cl_uint readFromFile(std::ifstream &inputFile, std::vector<FLOAT_TYPE> &demands)
{
    while (inputFile.good())
    {
        std::string char_value;
        getline (inputFile, char_value, ',' );
        demands.push_back(std::stof(char_value));  //May have problems with double
    }
    int num_station=demands.size();
    int remaining_size= pow(2, ceil(log(demands.size())/log(2)))-demands.size();
    demands.insert(demands.end(), remaining_size, 0);

    return num_station;
}

void generateRandom(std::vector<FLOAT_TYPE> &demands, cl_uint num_stations)
{
    const FLOAT_TYPE MULT_FACTOR=0.8;
    srand(time(nullptr));
    for (cl_uint k=0; k<num_stations; k++)
        demands.push_back(((FLOAT_TYPE)rand())*MULT_FACTOR/RAND_MAX);

    int remaining_size= pow(2, ceil(log(demands.size())/log(2)))-demands.size();
    demands.insert(demands.end(), remaining_size, 0);
}

void checkArrays(std::vector<FLOAT_TYPE> &arr1, std::vector<FLOAT_TYPE> &arr2)
{
    cl_uint fails=0;
    FLOAT_TYPE max_diff=0.0;
    if (arr1.size()==arr2.size())
    {
        for (cl_uint i=0; i<arr1.size(); i++)
        {
            FLOAT_TYPE diff=fabs(arr1[i]-arr2[i]);
            if (diff>ZERO_APPROX)
            {
                fails++;
            }
            if (diff>max_diff)
                max_diff=diff;
        }
        if (fails==0)
            std::cout<<"Arrays are (almost) equal."<<std::endl;
        else
            std::cout<<"ATTENTION: Residences with difference bigger than "<<ZERO_APPROX<<": "<<fails<<std::endl;
        std::cout<<"Max Difference: "<<max_diff<<std::endl;
    }
    else
    {
        std::cout<<"Arrays to be compared have different sizes! "<<std::endl;
    }
}

int main(int argc, char *argv[])
{
    using namespace std::chrono;
    std::ifstream inputFile;

    std::vector<FLOAT_TYPE> demands;
    cl_uint num_stations=NUM_STATIONS_DEFAULT;
    FLOAT_TYPE think_time=THINK_TIME_DEFAULT;
    cl_uint num_jobs=NUM_JOBS_DEFAULT;

    ////////////////////////////////////////////////////////////////////////////////
    //parsing input arguments
    ////////////////////////////////////////////////////////////////////////////////
    int next_option;
    //a string listing valid short options letters
    const char* const short_options = "n:z:d:k:h:";
    //an array describing valid long options
    const struct option long_options[] = { { "help", no_argument, NULL, 'h' }, //help
          { "demands", required_argument, NULL, 'd' }, //demands file
          { "think", required_argument, NULL, 'z' }, //think time
          { "jobs", required_argument, NULL, 'n' }, //number of jobs
          { "stations", required_argument, NULL, 'k' }, //Number of stations, if -d not specified
          { NULL, 0, NULL, 0 } /* Required at end of array.  */
    };

    next_option = getopt_long(argc, argv, short_options, long_options, NULL);
    while(next_option != -1)
    {
        switch (next_option)
        {
            case 'n':
                num_jobs=atoi(optarg);
                break;
            case 'z':
                think_time=(FLOAT_TYPE)atof(optarg);
                break;
            case 'k':
                num_stations=atoi(optarg);
                break;
            case 'd':
                inputFile.open(optarg);
                if (inputFile.is_open() && num_stations==NUM_STATIONS_DEFAULT){
                    num_stations=readFromFile(inputFile, demands);
                    inputFile.close();
                    break;
                }
            case '?':
            case 'h':
            default: /* Something else: unexpected.  */
                std::cerr << std::endl << "USAGE: " << argv[0] << std::endl;
                std::cerr << "[-n NUMBER_JOBS] - Specify Number of Jobs" << std::endl;
                std::cerr << "[-z THINK_TIME] - Specify a Think Time" << std::endl;
                std::cerr << "[-d FILEPATH] - Specify File Path of Demands" << std::endl;
                std::cerr << "---------------------------------------------------" << std::endl;
                std::cerr << "In the input text file, Demands values should be separated by comma, without spaces. " << std::endl;
                std::cerr << "The output text file with all Residence Times will be saved at path " <<FILENAME_RESIDENCE<< std::endl;
                std::cerr << std::endl;
                exit(EXIT_FAILURE);
        }
        next_option = getopt_long(argc, argv, short_options, long_options, NULL);
    }
    //Generating random stations, if necessary
    if(demands.size()==0){
        generateRandom(demands, num_stations);
    }

    FLOAT_TYPE throughput=0.0, sys_res=0.0;
    std::vector<FLOAT_TYPE> responseCPU(num_stations, 0);
    ///CPU computation
    high_resolution_clock::time_point start_time = high_resolution_clock::now();
    exactMVA(responseCPU, demands, num_stations, num_jobs, think_time);
    high_resolution_clock::time_point end_time = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(end_time - start_time);
    std::cout<<"Time required by CPU: "<<time_span.count()<<std::endl<<std::endl;

    //Printing Results
    for (cl_uint k=0; k<num_stations; k++)
        sys_res+=responseCPU[k];
    throughput=num_jobs/(think_time+sys_res);
    std::cout << "Global Throughput: "<<throughput<<std::endl;
    std::cout<<"System Response Time: "<<sys_res<<std::endl;
    std::cout<<"---------------------------------------------"<<std::endl;

    std::vector<FLOAT_TYPE> responseGPU(num_stations, 0);
    ///OpenCL computation
    start_time = high_resolution_clock::now();
    //Initializing OpenCL implementation
    read_cl_file();
    cl_initialization(demands.size());
    cl_load_prog(demands.size());
    cl_mem_init(demands.size());
    //Launching Execution
    cl_launch_kernel(responseGPU, demands, num_stations, num_jobs, think_time);
    end_time = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(end_time - start_time);
    //Cleaning Up memory Objects
    cl_clean_up();
    std::cout<<"Time required by GPU: "<<time_span.count()<<std::endl<<std::endl;

    //Printing Results
    sys_res=0.0;
    for (cl_uint k=0; k<num_stations; k++)
        sys_res+=responseGPU[k];
    throughput=num_jobs/(think_time+sys_res);
    std::cout << "Global Throughput: "<<throughput<<std::endl;
    std::cout<<"System Response Time: "<<sys_res<<std::endl<<std::endl;

    ///Check equality between CPU and GPU Arrays
    checkArrays(responseCPU, responseGPU);

    ///Saving Residence Times on File
    std::ofstream residence_file;
    residence_file.open(FILENAME_RESIDENCE);
    for (uint i=0; i<num_stations; i++)
    {
        residence_file<<responseCPU[i]<<",";
        if ((i+1)%10==0)
            residence_file<<std::endl;
    }

    residence_file<<std::endl<<std::endl;
    for (uint i=0; i<num_stations; i++)
    {
        residence_file<<responseGPU[i]<<",";
        if ((i+1)%10==0)
            residence_file<<std::endl;
    }
    std::cout<<"Residence Times saved in text file at: '"<<FILENAME_RESIDENCE<<"'"<<std::endl;
    residence_file.close();

    return 0;
}
