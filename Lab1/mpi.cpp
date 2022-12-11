#include <mpi.h>
#include <stdio.h>
#include <iostream>


int main(int argc, char* argv[])
{
	std::cout << "1";
	int rank, ranksize;
	double sumtime;
	double difftime;
	MPI_Init(&argc, &argv);
	double starttime = MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
	printf("Hello World! from process %d of %d\n", rank, ranksize);
	difftime = MPI_Wtime() - starttime;
	MPI_Reduce(&difftime, &sumtime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Finalize();
	
	if (rank == 0)
		printf("Time of working programm with %d threads = %f\n", ranksize, sumtime * 1000);
	return 0;
}
