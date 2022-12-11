#include "lab2.h"

using namespace std;

int main(int argc, char* argv[])
{
	int theads_count = argc > 1 ? atoi(argv[1]) : 1;
	omp_set_num_threads(theads_count);
	fstream atomic_out, critical_out, reduce_out;
	string path("results/");
    atomic_out.open(path + "atomic" + argv[1] + EXTENSION, ios::app);
    critical_out.open(path + "critical" + argv[1] + EXTENSION, ios::app);
    reduce_out.open(path + "reduce" + argv[1] + EXTENSION, ios::app);
	
	if (argc > 2 && string(argv[2]) == "q")
	{
		printf("qAtomic:\n");
		test_time(atomic_sum_q, atomic_out);
		printf("qCritical:\n");
		test_time(critical_sum_q, critical_out);
		printf("qReduce:\n");
		test_time(reduce_sum_q, reduce_out);
	} else
	{
		printf("Atomic:\n");
		test_time(atomic_sum, atomic_out);
		printf("Critical:\n");
		test_time(critical_sum, critical_out);
		printf("Reduce:\n");
		test_time(reduce_sum, reduce_out);
	}
	
	atomic_out.close(); critical_out.close(); reduce_out.close();
	return 0;
}

