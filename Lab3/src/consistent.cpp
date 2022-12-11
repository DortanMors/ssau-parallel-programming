// consistent.cpp
#include "lab2.h"

using namespace std;

int main(int argc, char* argv[])
{
	fstream fout;
	string path("results/");
    fout.open(path + "consistent" + argv[1] + EXTENSION, ios::app);
	
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

