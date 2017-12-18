#include <math.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "mpi.h"
#include "Lab2.h"
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds

#define Swap(x,y) {float* temp; temp = x; x = y; y = temp;}

using namespace std;

int main(int argc, char* argv[]) {
	int err_code = init(argc, argv);
	if (err_code != 0) {
		cout << "Error while MPI inizialization, error code: " << err_code << endl;
		return err_code;
	}

	sendcounts = new int[p_num];
	displs = new int[p_num];

	sendcountsA = new int[p_num];
	displsA = new int[p_num];

	readMtxs(argv[1]);
	int converged = 0;
	double totalTime = 0;
	for (int i = 0; i < 20; i++) {
		double t1 = MPI_Wtime();
		converged = parallel_jacobi(0.0001, 1000000);
		double t2 = MPI_Wtime();
		totalTime += t2 - t1;
	}
	
	collectData();


	if (my_rank == 0) {
		if (converged) {
			cout << "X:" << endl;
			write_X();
		}
		else {
			cout << "Error: solve is not converged";
		}
		cout << endl << (totalTime / 20) << endl;
	}


	MPI_Finalize();
	
	on_end();
	return 0;
}

int init(int argc, char *argv[]) {
	int err_code;
	if ((err_code = MPI_Init(&argc, &argv)) != 0)
	{
		return err_code;
	}
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p_num);
}

void readMtxs(char* filename) {
	if (my_rank == 0) {
		readRowsNum(filename);
	}

	MPI_Bcast(&rows_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(sendcounts, p_num, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(displs, p_num, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(sendcountsA, p_num, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(displsA, p_num, MPI_INT, 0, MPI_COMM_WORLD);

	int all_items_num = rows_num * rows_num;
	int for_one_proc_num = all_items_num / p_num;

	// temp stuff for sending and recieving 
	float *tempA = new float[all_items_num];
	float *tempRec = new float[all_items_num];
	float *tempB = new float[rows_num];
	float *tempX = new float[rows_num];

	init_main_arrays();

	if (my_rank == 0) {
		readMainData(filename, tempA, tempB, tempX);
	}

	MPI_Scatterv(tempA, sendcountsA, displsA, MPI_FLOAT,
		tempRec, sendcountsA[my_rank], MPI_FLOAT,
		0, MPI_COMM_WORLD);
	MPI_Scatterv(tempB, sendcounts, displs, MPI_FLOAT,
		mtx_B, sendcounts[my_rank], MPI_FLOAT,
		0, MPI_COMM_WORLD);
	MPI_Scatterv(tempX, sendcounts, displs, MPI_FLOAT,
		mtx_X, sendcounts[my_rank], MPI_FLOAT,
		0, MPI_COMM_WORLD);

	// from plain to matrix
	for (int i = 0; i < sendcountsA[my_rank] / (rows_num); i++) {
		for (int j = 0; j < rows_num; j++) {
			mtx_A[i][j] = tempRec[i * rows_num + j];
		}
	}

	delete[] tempA;
	delete[] tempRec;
	delete[] tempB;
	delete[] tempX;
}

void readRowsNum(char* filename) {
	try {
		ifstream input;
		int columns_all = 0;
		input.open(filename);
		input >> rows_num >> columns_all;
		input.close();
	}
	catch (ifstream::failure& e) {
		cout << "Error while reading file" << endl;
	}
	int n = rows_num / p_num;
	int for_first = rows_num % p_num;

	sendcounts[0] = n + for_first;
	sendcountsA[0] = n * rows_num + (rows_num * for_first);
	displs[0] = 0;
	displsA[0] = 0;

	for (int i = 1; i < p_num; i++) {
		sendcounts[i] = n;
		displs[i] = (n * (i - 1)) + (n + for_first);

		sendcountsA[i] = n * rows_num;
		displsA[i] = n * rows_num * i;
	}
}

void readMainData(char* filename, float* tempA, float* tempB, float* tempX) {
	try {
		ifstream input;
		input.open(filename);
		int columns_all = 0;
		input >> columns_all >> columns_all;

		for (int i = 0; i < rows_num; i++) {
			for (int j = 0; j < rows_num; j++) {
				input >> tempA[i * rows_num + j];
			}
			input >> tempB[i];
			tempX[i] = 0.0;
		}
		input.close();
	}
	catch (ifstream::failure& e) {
		cout << "Error while reading file" << endl;
	}
}
void init_main_arrays() {
	mtx_A = new float*[rows_num];
	for (int i = 0; i < rows_num; i++) {
		mtx_A[i] = new float[rows_num];
	}
	mtx_B = new float[rows_num];
	mtx_X = new float[rows_num];
}


int parallel_jacobi(float tol, int max_iter) {
	int i_local, i_global, j;
	int iter_num;
	float *x_old = new float[rows_num];
	float *x_new = new float[rows_num];

	for (int i = 0; i < rows_num; i++) {
		x_old[i] = 0.0;
		x_new[i] = 0.0;
	}

	iter_num = 0;
	do {
		/* Interchange x_old and x_new */
		Swap(x_old, x_new);
		for (i_local = 0; i_local < sendcounts[my_rank]; i_local++) {
			i_global = i_local + my_rank*sendcounts[my_rank];
			mtx_X[i_local] = mtx_B[i_local];

			for (j = 0; j < rows_num; j++) {
				if (j != i_global) {
					mtx_X[i_local] -= (mtx_A[i_local][j] * x_old[j]);
				}
			}

			mtx_X[i_local] /= mtx_A[i_local][i_global];
		}
		MPI_Allgatherv(mtx_X, sendcounts[my_rank], MPI_FLOAT, x_new, sendcounts,
			displs, MPI_FLOAT, MPI_COMM_WORLD);
		iter_num++;
	} while ((iter_num < max_iter) &&
		(Distance(x_new, x_old, rows_num) >= tol));

	if (Distance(x_new, x_old, rows_num) < tol) {
		return 1;
	}
	else {
		return 0;
	}
}

float Distance(float x[], float y[], int n) {
	int i;
	float sum = 0.0;

	for (i = 0; i < n; i++) {
		sum = sum + (x[i] - y[i])*(x[i] - y[i]);
	}
	return sqrt(sum);
}

void collectData() {
	float* temp = new float[rows_num];
	MPI_Gatherv(mtx_X, sendcounts[my_rank], MPI_FLOAT, temp, sendcounts, displs, MPI_FLOAT,
		0, MPI_COMM_WORLD);
	for (int i = 0; i < rows_num; i++) {
		mtx_X[i] = temp[i];
	}
	delete[] temp;

}

// Not interesting stuff
void on_end() {
	for (int i = 0; i < sendcounts[my_rank]; i++) {
		delete[] mtx_A[i];
	}
	delete[] mtx_A;
	delete[] mtx_B;
	delete[] mtx_X;
	delete[] sendcounts;
	delete[] displs;
	delete[] sendcountsA;
	delete[] displsA;
}

void write_A() {
	for (int i = 0; i < rows_num; i++) {
		for (int j = 0; j < rows_num; j++) {
			std::cout << mtx_A[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void write_B() {
	for (int j = 0; j < rows_num; j++) {
		std::cout << mtx_B[j] << std::endl;
	}
	std::cout << std::endl;
}

void write_X() {
	for (int j = 0; j < rows_num; j++) {
		std::cout << fixed << setprecision(9) << mtx_X[j] << std::endl;
	}
	std::cout << std::endl;
}
