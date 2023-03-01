// consistent.cpp

#include <stdlib.h>
#include <fstream>
#include <string>
#include <chrono>
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
  double average_time = 0.0;
  for (int t = 0; t < START_TIMES; ++t)
	{
    double* array = create_double_array(NMAX);
    auto start = std::chrono::system_clock::now();
    double sum = method(array, NMAX);
    auto end = std::chrono::system_clock::now();
    double result_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    average_time += result_time;
    out << result_time << std::endl;
    free(array);
    printf("Total Sum = %10.2f \n", sum);
    printf("TIME OF WORK IS %f microseconds\n", result_time);
  }
  average_time /= START_TIMES;
  out << "Average: " << average_time << endl;
}

int main(int argc, char* argv[])
{
  fstream fout;
  string path("results/");
  fout.open(path + "consistent" + ((argc > 1) ? argv[1] : "")  + EXTENSION, ios::app);
	
	if (argc > 1 && string(argv[1]) == "q")
	{
		printf("qConsistent:\n");
		test_time(sum_q, fout);
	} else
	{
		printf("Consistent:\n");
		test_time(sum, fout);
	}
	
	fout.close();
	return 0;
}

