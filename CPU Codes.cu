#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <device_functions.h>
#include "math.h"

//Matrix generation
void matgen(float* a, int lda, int m, int n);
//Branch Conductance matrix generation
void matgen_Gb(float* a, int lda, int m, int n);
//Matrix multiplication by CPU
void matmult(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int m, int n, int k);
//Matrix transpose
void mattanspose(float* a, int lda, float* b, int ldb, int m, int n);
//Matrix subtraction
void matsub(float* a, int lda, float* b, int ldb, float* c, int ldc, int m, int n);
//Print the result
void matprint(float* a, int Ida, int m, int n);
//Calculate Gn
void mat_cal_Gn(float* A, float* AT, float* Gb, float* AGb, float* Gn, int m, int n);
//Calculate Jn
void mat_cal_Jn(float* A, float* Es, float* AGb, float* AGbEs, float* Js, float* AJs, float* Jn, int m, int n);

void matgen(float* a, int lda, int m, int n)
{
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			scanf("%f", &a[i * lda + j]);
			//a[i * lda + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
		}
	}
}
void matgen_Gb(float* a, int lda, int m, int n)
{
	int i, j;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++) {
			if (i == j)
			{
				scanf("%f", &a[i * lda + j]);
				//a[i * lda + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
			}
			else
			{
				a[i * lda + j] = 0;
			}
		}
	}
}
void mattanspose(float* a, int lda, float* b, int ldb, int m, int n)
{
	int i, j;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			b[i * ldb + j] = a[j * lda + i];
		}
	}
}
void matmult(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int m, int n, int k) {
	int i, j, q;
	float t;
	for (i = 0; i < m; i++) {
		for (j = 0; j < k; j++) {
			t = 0;
			for (q = 0; q < n; q++) 
			{
				t += a[i * lda + q] * b[q * ldb + j];
			}
			c[i * ldc + j] = t;
		}
	}
}
void matsub(float* a, int lda, float* b, int ldb, float* c, int ldc, int m, int n)
{
	int i, j;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			c[i * ldc + j] = a[i * lda + j] - b[i * ldb + j];
		}
	}
}
void matprint(float* a, int lda, int m, int n)
{
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			printf("%f\t", a[i * lda + j]);
		}
		printf("\n");
	}

}
void mat_cal_Gn(float* A, float* AT, float* Gb, float* AGb, float* Gn, int m, int n)
{
	//transpose the incidience matrix
	mattanspose(A, n, AT, m, m, n);
	//printf("incidience matrix A is:\n");
	//matprint(A, n, m, n);
	//printf("The transpose matrix of A is:\n");
	//matprint(AT, m, n, m);
	//printf("The branch conductance matrix is:\n");
	//matprint(Gb, n, n, n);
	//Begin to calculate Gn
	matmult(A, n, Gb, n, AGb, n, m, n, n);
	//printf("The mediation matrix AGb is:\n");
	//matprint(AGb, n, m, n);
	matmult(AGb, n, AT, m, Gn, m, m, n, m);
	//printf("The node conductance matrix is:\n");
	//matprint(Gn, m, m, m);
}
void mat_cal_Jn(float* A, float* Es, float* AGb, float* AGbEs, float* Js, float* AJs, float* Jn, int m, int n)
{
	matmult(AGb, n, Es, 1, AGbEs, 1, m, n, 1);
	matmult(A, n, Js, 1, AJs, 1, m, n, 1);
	matsub(AGbEs, 1, AJs, 1, Jn, 1, m, 1);
	//printf("Mediation matrix AGbEs is:\n");
	//matprint(AGbEs, 1, m, 1);
	//printf("Mediation matrix AJs is:\n");
	////matprint(AJs, 1, m, 1);
	//printf("Node current source column vector is:\n");
	//matprint(Jn, 1, m, 1);
}

//LU factorization_CPU
float* A, * Es, * Js, * Gb, * Gn, * Jn, * Un, * Ib, * AT, * AGb, * AGbEs, * AJs;
double ** a, * b, * x, * y, ** L, ** U;
const int m = 7,n = 14;
unsigned int RANK = 4;

unsigned int makematrix()
{
	int r = m, c = m;
	//printf("Please input roww and column numbers of the matrix，space off：");

	a = (double**)malloc(sizeof(double*) * r);//Creat a pointer array，Assigns the address of the pointer array to a
	for (int i = 0; i < r; i++)
		a[i] = (double*)malloc(sizeof(double) * c);//Allot space to the second dimension
	for (int i = 0; i < r; i++) 
	{
		for (int j = 0; j < c; j++)
		{
			a[i][j] = 0.0;
		}
	}

	b = (double*)malloc(sizeof(double) * r);
	for (int i = 0; i < r; i++)
	{
		b[i] = 0.0;
	}

	x = (double*)malloc(sizeof(double) * c);
	for (int i = 0; i < c; i++)
	{
		x[i] = 0.0;
	}
	L = (double**)malloc(sizeof(double*) * r);//Creat a pointer array，Assigns the address of the pointer array to a
	for (int i = 0; i < r; i++)
		L[i] = (double*)malloc(sizeof(double) * c);//Allot space to the second dimension
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++)
			L[i][j] = 0.0;
	}
	U = (double**)malloc(sizeof(double*) * r);//Creat a pointer array，Assigns the address of the pointer array to a
	for (int i = 0; i < r; i++)
		U[i] = (double*)malloc(sizeof(double) * c);//Allot space to the second dimension
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++)
			U[i][j] = 0.0;
	}
	y = (double*)malloc(sizeof(double) * c);
	for (int i = 0; i < c; i++)
	{
		y[i] = 0.0;
	}
	return r;
}
void getmatrix(void)//Input the matrix and show it
{
	printf("Enter the coefficient matrix A in line from left to right, separating the elements with Spaces\n");
	for (int i = 0; i < RANK; i++)
	{
		for (int j = 0; j < RANK; j++)
		{
			//a[i][j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
			//scanf_s("%lf", &A[i][j]);
			a[i][j] = Gn[i * m + j];
		}
	}
	printf("The coefficient matrix is as follows\n");
	for (int i = 0; i < RANK; i++)
	{
		for (int j = 0; j < RANK; j++)
		{
			printf("%f\t", a[i][j]);
		}
		printf("\n");
	}
	printf("Enter the constant B from top to bottom, separated by Spaces\n");
	for (int i = 0; i < RANK; i++)
	{
		//b[i] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
		//scanf_s("%lf", &b[i]);
		b[i] = Jn[i];
	}
	printf("The constant sequence is as follows\n");
	for (int i = 0; i < RANK; i++)
	{
		printf("%f\n", b[i]);
	}
	printf("\n");
}
void LUrush_calculation(void)//LU factorization
{
	double get_add = 0.0;
	printf("The augmented matrix composed of A and B above is used to calculate the equations by LU decomposition method\n");

	for (int i = 0; i < RANK; i++)
	{
		U[0][i] = a[0][i];
		L[i][i] = 1;
	}
	for (int i = 1; i < RANK; i++)
	{
		L[i][0] = a[i][0] / U[0][0];
	}
	for (int i = 1; i < RANK; i++)
	{
		for (int j = 0; j < RANK; j++)
		{
			if (i <= j)
			{
				get_add = 0.0;
				for (int k = 0; k <= i - 1; k++) get_add = get_add + L[i][k] * U[k][j];
				U[i][j] = a[i][j] - get_add;
			}
			if (i < j)
			{
				get_add = 0.0;
				for (int k = 0; k <= i - 1; k++) get_add = get_add + L[j][k] * U[k][i];
				L[j][i] = (a[j][i] - get_add) / U[i][i];
			}
		}
	}
	/*printf("The L U matrix is\n");
	for (int i = 0; i < RANK; i++)
	{
		for (int j = 0; j < RANK; j++)
		{
			printf("%g\t", L[i][j]);
		}		printf("\t\t");
		for (int j = 0; j < RANK; j++)
		{
			printf("%g\t", U[i][j]);
		}		printf("\n");
	}*/
	//printf("solve y, and we can get：\n");
	y[0] = b[0];
	for (int i = 1; i < RANK; i++)
	{
		get_add = 0.0;
		for (int k = 0; k <= i - 1; k++) get_add = get_add + L[i][k] * y[k];
		y[i] = b[i] - get_add;
	}
	/*for (int i = 0; i < RANK; i++)
	{
		printf("y%d = %g\n", i + 1, y[i]);
	}*/
	//printf("solve y, and we can get：\n");
	x[RANK - 1] = y[RANK - 1] / U[RANK - 1][RANK - 1];
	for (int i = RANK - 2; i >= 0; i--)
	{
		get_add = 0.0;
		for (int k = i + 1; k <= RANK - 1; k++) get_add = get_add + U[i][k] * x[k];
		x[i] = (y[i] - get_add) / U[i][i];
	}
	for (int i = 0; i < RANK; i++)
	{
		printf("x%d = %g\n", i + 1, x[i]);
	}
}
int main()
{
	double start, end;
	//Define the matrix
	//matrix input by hand
	A = (float*)malloc(sizeof(float) * m * n);// A incidience matrix m*n
	Es = (float*)malloc(sizeof(float) * n * 1);// Es The branch voltage source column vector n*1
	Js = (float*)malloc(sizeof(float) * n * 1);// Js The branch current source column vector n*1
	Gb = (float*)malloc(sizeof(float) * n * n);// Gb Branch conductance matrix n*n diagonal matrix
	//A matrix used to construct an equation
	Gn = (float*)malloc(sizeof(float) * m * m);// Gn Node conductance matrix m*m
	Jn = (float*)malloc(sizeof(float) * m * 1);//Jn Node current source column vector m*1
	//The intermediate matrix
	AT = (float*)malloc(sizeof(float) * n * m);// A Transposed matrix n*m
	AGb = (float*)malloc(sizeof(float) * m * n);// A * Gb Mediation matrix
	AGbEs = (float*)malloc(sizeof(float) * m * 1);// A * Gb * Es Mediation matrix
	AJs = (float*)malloc(sizeof(float) * m * 1);// A * Js Mediation matrix
	//The result
	Un = (float*)malloc(sizeof(float) * m * 1);//Un Node voltage column vector m*1
	Ib = (float*)malloc(sizeof(float) * m * 1);//Ib Branch current column vector m*1
	//The matrix which needs to be input by hand
	printf("Please enter the incidience matrix A:\n");
	matgen(A, n, m, n);
	printf("Please enter the branch voltage source column vector Es:\n");
	matgen(Es, 1, n, 1);
	printf("Please enter the branch current source column vector Js:\n");
	matgen(Js, 1, n, 1);
	printf("Please enter the branch conductance matrix Gb:\n");
	matgen_Gb(Gb, n, n, n);
	//Timing begings
	start = clock();
	//Begin to calculate Gn
	mat_cal_Gn(A, AT,Gb, AGb,Gn,m,n);
	//Begin to calculate Jn
	mat_cal_Jn(A, Es, AGb, AGbEs, Js, AJs, Jn, m, n);
	//Transform Gn and Jn into two-dimensional matrix
	int i, j;
	/*for (i = 0; i < m; i++)
	{
		for (j = 0; j < m; j++)
		{
			a[i][j] = Gn[i * m + j];
		}
	}
	for (i = 0; i < m; i++)
	{
		b[i] = Jn[i];
	}*/
	RANK = makematrix();
	getmatrix();
	LUrush_calculation();
	end = clock();
	printf("Time consumed %f ms\n", end - start);
	return 0;
}



