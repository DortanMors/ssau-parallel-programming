#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int WORKING_TIMES = 12;


int main(int argc, char* argv[])
{
	int start_count = argc > 1 ? std::atoi(argv[1]) : 1;
	clock_t start = clock();
	for (int j = 0; j < WORKING_TIMES; ++j)
	{
		for (int i = 0; i < start_count; ++i)
		{
			printf("Hello World! from start %d of %d\n", i, start_count);
		}
	}
	clock_t stop = clock();
	double elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC / WORKING_TIMES;
	printf("Programm working time: %f\n", elapsed);
	return 0;
}
