#include <math.h>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <string>
#include "mpi.h"
#include "common.h"
#define NMAX 6000000

using namespace std;

int main(int argc, char* argv[]) {
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
	if (ProcRank == 0) {
		x = create_double_array(NMAX);
	}
	// подготовка данных, рассылка 0-ым процессом по всем остальным
	double starttime = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
	MPI_Scatter(x, numPerProc, MPI_DOUBLE, subArray, numPerProc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (argc > 1 && string(argv[1]) == "q") {
		for ( int k = 0; k < numPerProc; ++k ) {
			ProcSum += sum_with_q(subArray[k]);
		}
	} else {
		for ( int k = 0; k < numPerProc; ++k ) {
			ProcSum += subArray[k];
		}
	}
	
	if (ProcRank == 0)
	{
		TotalSum = ProcSum;
		for (int i = 1; i < ProcNum; ++i) {
			MPI_Recv(&ProcSum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &Status);
			TotalSum += ProcSum;
		}
	} else {
		MPI_Send(&ProcSum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);
    difftime = (MPI_Wtime() - starttime) * 1000 * 1000;
	MPI_Reduce(&difftime, &sumtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if (ProcRank == 0) {
		ofstream fout;
		string path("results/");
		fout.open(path + "point_to_point" + to_string(ProcNum) + ".txt", ios::app);
		fout << sumtime << endl;
		fout.close();
		printf("Total Sum = %10.2f\n", TotalSum);
		printf("TIME OF WORK IS %.2f microseconds\n", sumtime);
	}
	MPI_Finalize();
	return 0;
}
