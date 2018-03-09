float:
	g++ -std=c++11 -w main.cpp -o mva -l OpenCL
double:
	g++ -std=c++11 -w -D DOUBLE main.cpp -o mva -l OpenCL
