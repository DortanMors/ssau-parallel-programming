// lab3
#include <stdlib.h>
#include "mpi.h"
#include <stdio.h>
#include <iostream>
#define NMAX 6000000

int main(int argc, char* argv[]) {
	double* a = nullptr, *b = nullptr, *sum = nullptr;
	double* a_loc = nullptr, *b_loc = nullptr, *sum_loc = nullptr;
	int ProcRank, ProcNum;
	MPI_Status Status;
	double st_time, end_time;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	int count = NMAX / ProcNum;

	if (ProcRank == 0) {
		a = (double*)malloc(NMAX * sizeof(double));
		b = (double*)malloc(NMAX * sizeof(double));
		sum = (double*)malloc(NMAX * sizeof(double));
		for (int i = 0; i < NMAX; i++) {
			a[i] = rand() % 10 + 1;
			b[i] = rand() % 10 + 11;
		}
	}
	a_loc = (double*)malloc(count * sizeof(double));
	sum_loc = (double*)malloc(count * sizeof(double));
	b_loc = (double*)malloc(count * sizeof(double));
	if (NMAX % ProcNum == 0) {
		// кратный случай
		MPI_Scatter(a, count, MPI_DOUBLE, a_loc, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatter(b, count, MPI_DOUBLE, b_loc, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (ProcRank == 0) {
			int time = 0;
			st_time = MPI_Wtime();
		}
		for (int i = 0; i < count; i++) {
			// получение локальной суммы векторов
			sum_loc[i] = a_loc[i] + b_loc[i];
		}
		// сборка результата 0-ым процессом
		MPI_Gather(sum_loc, count, MPI_DOUBLE, sum, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	} else {
	    // не кратный случай
		int* counts = (int*)malloc(sizeof(int) * ProcNum);
		int* displs = (int*)malloc(sizeof(int) * ProcNum);
		for (int i = 0; i < ProcNum - 1; i++) {
			counts[i] = count; // NMAX / ProcNum
			displs[i] = count * i;
		}
		counts[ProcNum - 1] = NMAX - (ProcNum - 1) * count;
		displs[ProcNum - 1] = count * (ProcNum - 1);
		if (ProcRank == ProcNum - 1) {
			count = NMAX - (ProcNum - 1) * count;
		}
		MPI_Scatterv(a, counts, displs, MPI_DOUBLE, a_loc, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatterv(b, counts, displs, MPI_DOUBLE, b_loc, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (ProcRank == 0) {
			int time = 0;
			st_time = MPI_Wtime();
		}
		for (int i = 0; i < count; i++) {
			sum_loc[i] = a_loc[i] + b_loc[i];
		}
		// сборка результата
		MPI_Gatherv(sum_loc, count, MPI_DOUBLE, sum, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		free(counts);
		free(displs);
	}
	if (ProcRank == 0) {
		end_time = MPI_Wtime();
		printf("\nTIME OF WORK IS %f ", end_time - st_time);
		free(a);
		free(b);
		free(sum);
	}
	MPI_Finalize();
}
