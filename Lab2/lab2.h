// lab2.h
#include <omp.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include "common.h"
#include "stdio.h"
#define NMAX 6000000
#define START_TIMES 12
#define EXTENSION ".txt"

using namespace std;

double sum(double* array, long n)
{
	int i;
	double sum = 0;
	for (i = 0; i < n; ++i)
	{
		sum += array[i];
	}
	return sum;
}

double sum_q(double* array, long n)
{
	int i;
	double reduce_sum = 0;
	for (i = 0; i < NMAX; i++)
	{
		reduce_sum += sum_with_q(array[i]);
	}
	return reduce_sum;
}

void test_time(double (*method)(double*, long), std::ostream& out)
{
	for (int t = 0; t < START_TIMES; ++t)
	{
		double* array = create_double_array(NMAX);
		double start = omp_get_wtime();
		double sum = method(array, NMAX);
		double result_time = (omp_get_wtime() - start) * 1000; // time in ms
		out << result_time << std::endl;
		free(array);
		printf("Total Sum = %10.2f \n", sum);
		printf("TIME OF WORK IS %f ms\n", result_time);
	}
}

double atomic_sum(double* array, long n)
{
	int i;
	double sum = 0;
	#pragma omp parallel for shared(array) private(i)
	for (i = 0; i < n; ++i)
	{
		#pragma omp atomic update
		sum += array[i];
	}
	return sum;
}


double critical_sum(double* array, long n)
{
	int i;
	double sum = 0;
	#pragma omp parallel for shared(array) private(i)
	for (i = 0; i < NMAX; i++)
	{
		#pragma omp critical
		sum += array[i];
	}
	return sum;
}


double reduce_sum(double* array, long n)
{
	int i;
	double reduce_sum = 0;
	#pragma omp parallel for shared(array) reduction(+: reduce_sum) private(i)
	for (i = 0; i < NMAX; i++)
	{
		reduce_sum += array[i];
	}
	return reduce_sum;
}

double atomic_sum_q(double* array, long n)
{
	int i;
	double sum = 0;
	#pragma omp parallel for shared(array) private(i)
	for (i = 0; i < n; ++i)
	{
		#pragma omp atomic update
		sum += sum_with_q(array[i]);
	}
	return sum;
}


double critical_sum_q(double* array, long n)
{
	int i;
	double sum = 0;
	#pragma omp parallel for shared(array) private(i)
	for (i = 0; i < NMAX; i++)
	{
		#pragma omp critical
		sum += sum_with_q(array[i]);
	}
	return sum;
}


double reduce_sum_q(double* array, long n)
{
	int i;
	double reduce_sum = 0;
	#pragma omp parallel for shared(array) reduction(+: reduce_sum) private(i)
	for (i = 0; i < NMAX; i++)
	{
		reduce_sum += sum_with_q(array[i]);
	}
	return reduce_sum;
}

