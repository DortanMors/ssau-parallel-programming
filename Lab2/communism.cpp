#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include "common.h"
#include "mpi.h"
#define NMAX 6000000

using namespace std;

int main(int argc, char* argv[])
{
	double ProcSum = 0.0;
	int ProcRank, ProcNum, N=NMAX, i;
	MPI_Status Status;

	double TotalSum;
	double sumtime;
	double difftime;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD,&ProcRank);
	int numPerProc = NMAX / ProcNum;
	double* x;
	double* subArray = (double*)malloc(sizeof(double) * numPerProc);
	if (ProcRank==0)
	{
		x = create_double_array(NMAX);
	}
	// подготовка данных, рассылка 0-ым процессом по всем остальным
	double starttime = MPI_Wtime();
	MPI_Scatter(x, numPerProc, MPI_DOUBLE, subArray, numPerProc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (argc > 1 && string(argv[1]) == "q")
	{
		for ( int k = 0; k < numPerProc; ++k )
		{
			ProcSum += sum_with_q(subArray[k]);
		}
	} else
	{
		for ( int k = 0; k < numPerProc; ++k )
		{
			ProcSum += subArray[k];
		}
	}
	MPI_Reduce(&ProcSum, &TotalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	difftime = MPI_Wtime() - starttime ;

	MPI_Reduce(&difftime, &sumtime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (ProcRank == 0)
	{
		fstream fout;
		string path("results/");
		fout.open(path + "collection" + to_string(ProcNum) + ".txt", ios::app);
		fout << sumtime / ProcNum * 1000 << endl;
		fout.close();
		printf("Total Sum = %10.2f \n", TotalSum);
		printf("TIME OF WORK IS %f \n", sumtime);
	}
	MPI_Finalize();
	return 0;
}
