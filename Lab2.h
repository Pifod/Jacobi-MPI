#ifndef LAB_2
#define LAB_2

int my_rank;
int p_num;
// int my_number_rows;
int* sendcounts;
int* displs;
int* sendcountsA;
int* displsA;

int init(int argc, char* argv[]);
void init_main_arrays();


int rows_num;
int max_iter;
float **mtx_A;
float *mtx_B;
float *mtx_X;

void readMtxs(char* filename);
void readRowsNum(char* filename);
void readMainData(char* filename, float* tempA, float* tempB, float* tempX);
void on_end();

void write_A();
void write_B();
void write_X();
void collectData();

int parallel_jacobi(float, int);
float Distance(float x[], float y[], int n);

#endif