#include "Intestazione.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>

void serialize(int N, int* matrix, int* serMatrix, int Nsot)
{
	int i, j;
	int k = 0;
	int rDisp = 0, cDisp = 0;
	while (k < N*N) {
		for (i = rDisp; i < rDisp + Nsot; i++)
			for (j = cDisp; j < cDisp + Nsot; j++) {
				serMatrix[k] = matrix[i*N + j];
				k++;
			}
		cDisp = (cDisp + Nsot) % N;
		if (cDisp == 0)
			rDisp += Nsot;
	}
}

void apsp1D(int id, int p, int* dists, int n) {
	int i, j, k;
	int offset;
	int root;
	int* tmp;
	int size = n / p;
	tmp = (int*)malloc(n * sizeof(int));
	for (k = 0; k < n; k++) {
		root = 0;
		while (k >= root*(n / p) + (n / p)) {
			root++;
		}
		if (root == id) {
			offset = k - id*(n / p); //+BLOW_LOW(id, p, n);
			for (j = 0; j < n; j++) {
				tmp[j] = dists[offset*n + j];
			}
		}
		MPI_Bcast(tmp, n, MPI_INT, root, MPI_COMM_WORLD);
		for (i = 0; i < n / p; i++)
			for (j = 0; j < n; j++) {
				//printf("P=%d K=%d i=%d j=%d\n", id, k, i, j);
				//printf("old=%d new=%d\n\n", dists[i*n + j], dists[i*n + k] + tmp[j]);
				dists[i*n + j] = (dists[i*n + j] < dists[i*n + k] + tmp[j]) ?
					dists[i*n + j] : dists[i*n + k] + tmp[j];
			}
	}
	free(tmp);
}

void apsp2D(int id, int p, int* dists, int n) {
	int i;
	int size = n/sqrt(p);
	int nproc = p, rank = id, rank_griglia;
	int dim[2], periodicita[2], coord[2];

	MPI_Status status;                              // crea la variabile "status per la comunicazione"  
	MPI_Comm griglia;                               // variabile associata al nuovo comunicatore      

	dim[0] = dim[1] = sqrt(p);                            // dimensione della griglia logica n x n 
	periodicita[0] = periodicita[1] = 1;            // definisco cicliche sia sull'asse x che sull'asse y

													// crea nuovo communicator, chiamato griglia
	MPI_Cart_create(MPI_COMM_WORLD, 2, dim, periodicita, 1, &griglia);
	// trova il mio rank nel communicator griglia
	MPI_Comm_rank(griglia, &rank_griglia);
	// trova le mie coordinate in questo comm 
	MPI_Cart_coords(griglia, rank_griglia, 2, coord);
	//printf("Process %d %d started\n", coord[0], coord[1]);
	int remain_r[] = { 1, 0 };
	int remain_c[] = { 0, 1 };
	
	MPI_Comm row;
	//MPI_Cart_sub(griglia, remain_r, &row);
	MPI_Comm_split(MPI_COMM_WORLD, coord[0], rank, &row);
	MPI_Comm col;
	//MPI_Cart_sub(griglia, remain_r, &col);
	MPI_Comm_split(MPI_COMM_WORLD, coord[1], rank, &col);
	int rank_r, rank_c;

	MPI_Comm_rank(row, &rank_r);
	MPI_Comm_rank(col, &rank_c);
	int* tmp_r = (int*) malloc(size*sizeof(int));
	int* tmp_c = (int*) malloc(size * sizeof(int));

	for (int k = 0; k < n; k++) {
		int root_r = 0;
		int root_c = 0;
		while (k >= root_r*size + size) {
			root_r++;
			root_c++;
		}
		if (root_c == rank_c) {
			int offset = k - rank_c*size; //+BLOW_LOW(id, p, n);
			for (int j = 0; j < size; j++) {
				tmp_r[j] = dists[offset*size + j];
			}
		}
		if (root_r == rank_r) {
			int offset = k - rank_r*size; //+BLOW_LOW(id, p, n);
			
			for (int j = 0; j < size; j++) {
				tmp_c[j] = dists[j*size + offset];
			}
		}
		MPI_Bcast(tmp_r, size, MPI_INT, root_r, col);
		MPI_Bcast(tmp_c, size, MPI_INT, root_c, row);
		MPI_Barrier(MPI_COMM_WORLD);

		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++) {
				dists[i*size + j] = (dists[i*size + j] < tmp_r[j] + tmp_c[i]) ?
					dists[i*size + j] : tmp_r[j] + tmp_c[i];
			}
	}
}

int main_mpi(int argc, char** argv) {
	int dim = 1;
	std::mt19937 rng;
	rng.seed(std::random_device()());
	int rank, size;
	std::chrono::steady_clock::time_point begin;
	std::chrono::steady_clock::time_point end;
	std::ofstream myfile;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if(rank == 0) myfile.open("MPIDati.txt", std::fstream::app);
	for (int N = 2048; N <= 4000; N *= 2) {
		if (size > N) continue;
		int  i, j;
		std::uniform_int_distribution<std::mt19937::result_type> dist(1, N);
		int* dists;
		dists = new int[N*N];
		int* dists2;
		dists2 = (int*)malloc(N*N * sizeof(int));
		if (rank == 0) {
			
			for (int i = 0; i < N; i++)
				for (int j = 0; j < N; j++)
					if (i == j) dists[i*N + j] = 0;
					else dists[i*N + j] = dist(rng);
		}
		if (rank == 0) {
			if (dim == 2) {
				serialize(N, dists, dists2, sqrt(size));
				dists = dists2;
			}
		}
		if (dim == 1) {
			int rows = N / size;
			if (rank == 0) begin = std::chrono::steady_clock::now();
			if (rank == 0)
				MPI_Scatter(dists, N*rows, MPI_INT, MPI_IN_PLACE, N*rows, MPI_INT, 0, MPI_COMM_WORLD);
			else
				MPI_Scatter(dists, N*rows, MPI_INT, dists, N*rows, MPI_INT, 0, MPI_COMM_WORLD);
			apsp1D(rank, size, dists, N);
			if (rank == 0)
				MPI_Gather(MPI_IN_PLACE, N*rows, MPI_INT, dists, N*rows, MPI_INT, 0, MPI_COMM_WORLD);
			else
				MPI_Gather(dists, N*rows, MPI_INT, dists, N*rows, MPI_INT, 0, MPI_COMM_WORLD);
			if (rank == 0) end = std::chrono::steady_clock::now();
			if (rank == 0) {
				myfile << "Time difference(" << N << ", " << size << ") = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
			}
		}
		else if (dim == 2) {
			int sub = N / sqrt(size);
			if (rank == 0) begin = std::chrono::steady_clock::now();
			if (rank == 0)
				MPI_Scatter(dists, sub*sub, MPI_INT, MPI_IN_PLACE, sub*sub, MPI_INT, 0, MPI_COMM_WORLD);
			else
				MPI_Scatter(dists, sub*sub, MPI_INT, dists, sub*sub, MPI_INT, 0, MPI_COMM_WORLD);
			apsp2D(rank, size, dists, N);
			if (rank == 0)
				MPI_Gather(MPI_IN_PLACE, sub*sub, MPI_INT, dists, sub*sub, MPI_INT, 0, MPI_COMM_WORLD);
			else
				MPI_Gather(dists, sub*sub, MPI_INT, dists, sub*sub, MPI_INT, 0, MPI_COMM_WORLD);
			if (rank == 0) end = std::chrono::steady_clock::now();
			if (rank == 0) {
				myfile << "Time difference(" << N << ", " << size << " = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	if (rank == 0) myfile.close();
	MPI_Finalize();
	return 0;
}