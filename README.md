# ExactMVA_OpenCL
OpenCL implementation of the exact Mean Value Analysis algorithm.

For this first release, the application first computes MVA with a simple c++ algorithm on the CPU. Then uses the OpenCL acceleretad algorithm and compare execution times and results.

Requirements:
An OpenCL Platform is required to be already installed, OpenCL headers have to be present. To check that, you can use the "clinfo" bash tool. You can also check if your OpenCL device supports Double-Precision floating point computations.

How to compile:
Use command "make double" if your OpenCL device supports double precision floating point data and computation. If your device do not support double, use the command "make float".

How to launch application: Type command "./mva" followed by one or more of the following options: 
-d [DEMANDS_FILE_PATH] 
-n [number of jobs] 
-z [think time] 
-k [number of stations] -> can be used only if "-d" option is not already used, in order to specify how many stations have to be considered. Demands of those stations are generated randomly.

Example: "./mva -d demands.txt -n 200 -z 1.5" 
If you do not have a demands file as source and you eant to try random demands, use: "./mva -k 350 -n 5000 -z 0.5"

Demands file has to be a text file containing demands values separated with comma, without spaces. Last value should not have a comma. You can look at the pre-loaded file "./demands.txt" in the folder of the project as an example. 
The number of stations is computed automaticcaly from the file, it is not required to be specified by "-k" attribute.

As result, a file containing all the Residence times of stations is produced and saved as "./residences.txt".
