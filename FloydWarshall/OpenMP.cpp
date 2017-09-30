#include "Intestazione.h"
#include <iostream>
#include <random>
#include <omp.h>
#include <chrono>
#include <vector>

int min(int a, int b) {
	return (a < b) ? a : b;
}

void floydwarshall1d(int* mat, size_t n, int threads){
#pragma omp parallel num_threads(threads)
	for (int k = 0; k < n; ++k) {
#pragma omp for
		for (int i = 0; i < n; ++i) {
			auto v = mat[i*n + k];
			for (int j = 0; j < n; ++j) {
				auto val = v + mat[k*n + j];
				if (mat[i*n + j] > val) {
					mat[i*n + j] = val;
				}
			}
		}
	}
}

void floydwarshall2d(int* mat, size_t n, int threads) {
#pragma omp parallel num_threads(threads)
	for (int k = 0; k < n; ++k) {
#pragma omp for 
		for (int i = 0; i < n*n; ++i) {
			int r = i / n;
			int c = i % n;
			auto v = mat[r*n + k];
			auto val = v + mat[k*n + c];
			if (mat[i] > val) {
				mat[i] = val;
			}
		}
	}
}

void floydwarshallseq(int* mat, size_t n, int threads = 0)
{
	for (int k = 0; k < n; ++k) {
		for (int i = 0; i < n; ++i) {
			auto v = mat[i*n + k];
			for (int j = 0; j < n; ++j) {
				auto val = v + mat[k*n + j];
				if (mat[i*n + j] > val) {
					mat[i*n + j] = val;
				}
			}
		}
	}
}


void main_openmp() {
	for (int n = 2; n <= 2048; n *= 2) {
		for (int p = 32*32; p <= 32*32; p++) {
			std::mt19937 rng;
			rng.seed(std::random_device()());
			std::uniform_int_distribution<std::mt19937::result_type> dist(1, n);
			int* graph = new int[n*n];
			int* graph2 = new int[n*n];
			for (int i = 0; i < n; i++)
				for (int j = 0; j < n; j++) {
					if (i == j) graph[i*n + j] = 0;
					else graph[i*n + j] = dist(rng);
					graph2[i*n + j] = graph[i*n + j];
				}

			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

			floydwarshall1d(graph, n, p);

			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
		}
	}
	int c;
	std::cin >> c;
}