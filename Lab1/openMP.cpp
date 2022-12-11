#include<omp.h>
#include<stdlib.h>
#include<stdio.h>
#include <iostream>


int main(int argc, char* argv[])
{
	std::cout << "1";
	int theads_count = argc > 1 ? std::atoi(argv[1]) : 1;
	int thead_num;
	omp_set_num_threads(theads_count);
	double start = omp_get_wtime();
	#pragma omp parallel  private(thead_num)
	{
		thead_num = omp_get_thread_num();
		printf("Hello World! from thread %d of %d threads \n", thead_num, theads_count);
	}
	double end = omp_get_wtime() - start;
	printf("Time of working programm with %d threads = %f \n", thead_num, end * 1000);
	return 0;
}
