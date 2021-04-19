#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <device_functions.h>
using namespace std;
#define THREAD_NUM 256
#define BLOCK_NUM 16
const int m =7;
const int n = 14;
float a[m][m];
float b[m];
float GnUn[m];
void printDeviceProp(const cudaDeviceProp& prop)
{
	printf("Device Name : %s.\n", prop.name);
	printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
	printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
	printf("regsPerBlock : %d.\n", prop.regsPerBlock);
	printf("warpSize : %d.\n", prop.warpSize);
	printf("memPitch : %d.\n", prop.memPitch);
	printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("totalConstMem : %d.\n", prop.totalConstMem);
	printf("major.minor : %d.%d.\n", prop.major, prop.minor);
	printf("clockRate : %d.\n", prop.clockRate);
	printf("textureAlignment : %d.\n", prop.textureAlignment);
	printf("deviceOverlap : %d.\n", prop.deviceOverlap);
	printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}
//cuda
bool InitCUDA();
// Generate matrix a and b
void matgen(float* a, int lda, int m, int n);
//CUDA configuration functions, including memory allocation, number transfer, call kernel function calculation, etc
clock_t matmultCUDA(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int m, int n, int k);//m*n矩阵和n*k矩阵
//进行GPU计算的核函数
__global__ static void matMultCUDA(const float* a, size_t lda, const float* b, size_t ldb, float* c, size_t ldc, int m, int n, int k);
//Matrix generation
void matgen(float* a, int lda, int m, int n);
//Branch Conductance matrix generation
void matgen_Gb(float* a, int lda, int m, int n);
//Matrix multiplication by CPU
void mattanspose(float* a, int lda, float* b, int ldb, int m, int n);
//Matrix subtraction
void matsub(float* a, int lda, float* b, int ldb, float* c, int ldc, int m, int n);
//Print the result
void matprint(float* a, int Ida, int m, int n);
//Calculate Gn
void mat_cal_Gn(float* A, float* AT, float* Gb, float* AGb, float* Gn, int m, int n);
//Calculate Jn
void mat_cal_Jn(float* A, float* Es, float* AGb, float* AGbEs, float* Js, float* AJs, float* Jn, int m, int n);
int matrix_sloving();

bool InitCUDA()
{
	int count;
	//get the information of the GPU
	cudaGetDeviceCount(&count);

	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;

	for (i = 0; i < count; i++) {

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		//Print device information
		printDeviceProp(prop);

		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}

	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}
clock_t matmultCUDA(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int m, int n, int k)
{
	float* ac, * bc, * cc;
	clock_t start, end;

	start = clock();

	size_t pitch_a, pitch_b, pitch_c;
	int newn = ((m+n+k + BLOCK_NUM - 1) / BLOCK_NUM) * BLOCK_NUM;
	cudaMallocPitch((void**)&ac, &pitch_a, sizeof(float) * newn, newn);
	cudaMallocPitch((void**)&bc, &pitch_b, sizeof(float) * newn, newn);
	cudaMallocPitch((void**)&cc, &pitch_c, sizeof(float) * newn, newn);
	cudaMemset(ac, 0, pitch_a * newn);
	cudaMemset(bc, 0, pitch_b * newn);
	cudaMemcpy2D(ac, pitch_a, a, sizeof(float) * lda, sizeof(float) * n, m, cudaMemcpyHostToDevice);
	cudaMemcpy2D(bc, pitch_b, b, sizeof(float) * ldb, sizeof(float) * k, n, cudaMemcpyHostToDevice);
	int bx = (m + n + k + BLOCK_NUM - 1) / BLOCK_NUM;
	dim3 blocks(bx, bx);
	dim3 threads(BLOCK_NUM, BLOCK_NUM);
	matMultCUDA << <blocks, threads >> > (ac, pitch_a / sizeof(float), bc, pitch_b / sizeof(float), cc, pitch_c / sizeof(float), m, n, k);

	cudaMemcpy2D(c, sizeof(float) * ldc, cc, pitch_c, sizeof(float) * k, m, cudaMemcpyDeviceToHost);

	cudaError_t error = cudaGetLastError();
	printf("CUDA error: %s\n", cudaGetErrorString(error));

	cudaFree(ac);
	cudaFree(bc);
	cudaFree(cc);

	end = clock();

	return end - start;
}
__global__ static void matMultCUDA(const float* a, size_t lda, const float* b, size_t ldb, float* c, size_t ldc, int m, int n, int k)
{
	__shared__ float matA[BLOCK_NUM][BLOCK_NUM];
	__shared__ float matB[BLOCK_NUM][BLOCK_NUM];

	const int tidc = threadIdx.x;//thread column
	const int tidr = threadIdx.y;//thread row
	const int bidc = blockIdx.x * BLOCK_NUM;//block column
	const int bidr = blockIdx.y * BLOCK_NUM;//block row

	int i, j;
	float results = 0;
	float comp = 0;

	for (j = 0; j < k+n+m; j += BLOCK_NUM)
	{
		matA[tidr][tidc] = a[(tidr + bidr) * lda + tidc + j];
		matB[tidr][tidc] = b[(tidr + j) * ldb + tidc + bidc];

		//Shared memory data synchronization within a block
		__syncthreads();

		for (i = 0; i < BLOCK_NUM; i++)
		{
			results += matA[tidr][i] * matB[i][tidc];
		}
		__syncthreads();
	}
	c[(tidr + bidr) * ldc + tidc + bidc] = results;
}
__global__ void ComputeLk(float* A, int k, int n)
{
	int bx, gx, BX, GX, tx, mx, x, num;
	GX = gridDim.x;
	BX = blockDim.x;
	bx = threadIdx.x;
	gx = blockIdx.x;
	tx = gx * BX + bx;
	num = (n - k - 1 + BX * GX - 1) / (BX * GX);
	for (x = 0; x < num; x++)
	{
		mx = x * BX * GX + tx + k + 1;
		if (mx >= n)
			break;
		A[mx * n + k] = A[mx * n + k] / A[k * n + k];
	}

}
__global__ void ComputeL(float* A, int k, int n)
{
	int bx, gx, gy, BX, GX, tx, mx, x, numx, numy, BY, GY, ty, my, by, y;
	GX = gridDim.x; GY = gridDim.y;
	BX = blockDim.x; BY = blockDim.y;
	bx = threadIdx.x; by = threadIdx.y;
	gx = blockIdx.x; gy = blockIdx.y;
	tx = gx * BX + bx; ty = gy * BY + by;
	numx = (n - k - 1 + BX * GX - 1) / (BX * GX);
	numy = (n - k - 1 + BY * GY - 1) / (BY * GY);
	for (x = 0; x < numx; x++)
		for (y = 0; y < numy; y++)
		{
			mx = x * BX * GX + tx + k + 1;
			my = y * BY * GY + ty + k + 1;
			if (mx >= n)
				break;
			if (my >= n)
				continue;
			A[mx * n + my] = A[mx * n + my] - A[mx * n + k] * A[k * n + my];

		}

}
__global__ void L(float* A, float* b, int k, int n)
{
	int bx, gx, BX, GX, tx, mx, x, num;
	GX = gridDim.x;
	BX = blockDim.x;
	bx = threadIdx.x;
	gx = blockIdx.x;
	tx = gx * BX + bx;
	num = (n - k - 1 + BX * GX - 1) / (BX * GX);
	for (x = 0; x < num; x++)
	{
		mx = x * BX * GX + tx + k + 1;
		if (mx >= n)
			break;
		b[mx] = b[mx] - A[mx * n + k] * b[k];
	}
}
__global__ void ComputeU(float* A, float* b, int k, int n)
{
	int bx, gx, BX, GX, tx, mx, x, num;
	GX = gridDim.x;
	BX = blockDim.x;
	bx = threadIdx.x;
	gx = blockIdx.x;
	tx = gx * BX + bx;
	num = (k + BX * GX - 1) / (BX * GX);
	for (x = 0; x < num; x++)
	{
		mx = x * BX * GX + tx;
		if (mx > k - 1)
			break;
		b[mx] = b[mx] - b[k] * A[mx * n + k] / A[k * n + k];
	}
}

void matprint(float* a, int lda, int m, int n)
{
	int i, j;

	for (i = 0; i < m; i++) 
	{
		for (j = 0; j < n; j++) 
		{
			printf("%f\t", a[i * lda + j]);
		}
		printf("\n");
	}

}
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
	matmultCUDA(A, n, Gb, n, AGb, n, m, n, n);
	//printf("The mediation matrix AGb is:\n");
	//matprint(AGb, n, m, n);
	matmultCUDA(AGb, n, AT, m, Gn, m, m, n, m);
	//printf("The node conductance matrix is:\n");
	//matprint(Gn, m, m, m);
}
void mat_cal_Jn(float* A, float* Es, float* AGb, float* AGbEs, float* Js, float* AJs, float* Jn, int m, int n)
{
	matmultCUDA(AGb, n, Es, 1, AGbEs, 1, m, n, 1);
	matmultCUDA(A, n, Js, 1, AJs, 1, m, n, 1);
	matsub(AGbEs, 1, AJs, 1, Jn, 1, m, 1);
	//printf("Mediation matrix AGbEs is:\n");
	//matprint(AGbEs, 1, m, 1);
	//printf("Mediation matrix AJs is:\n");
	//matprint(AJs, 1, m, 1);
	//printf("Node current source column vector is:\n");
	//matprint(Jn, 1, m, 1);
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

int matrix_sloving()
{
	clock_t start, finish;
	//srand(time(0));
	int k, i, j; float* A_dev; float* b_dev; double alltime;

	cudaMalloc((void**)&A_dev, m * m * sizeof(float));
	cudaMalloc((void**)&b_dev, m * sizeof(float));
	cudaMemcpy(A_dev, (float*)a, m * m * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(b_dev, (float*)b, m * sizeof(float), cudaMemcpyHostToDevice);

	start = clock();
	for (k = 0; k < m - 1; k++)
	{
		ComputeLk << <BLOCK_NUM, THREAD_NUM, 0 >> > (A_dev, k, m);
		ComputeL << <BLOCK_NUM, THREAD_NUM, 0 >> > (A_dev, k, m);
	}
	for (k = 0; k < m - 1; k++)
	{
		L << <BLOCK_NUM, THREAD_NUM, 0 >> > (A_dev, b_dev, k, m);
	}
	for (k = m - 1; k > 0; k--)
	{
		ComputeU << <BLOCK_NUM, THREAD_NUM, 0 >> > (A_dev, b_dev, k, m);
	}
	finish = clock();
	cudaMemcpy(a, A_dev, m * m * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(b, b_dev, m * sizeof(float), cudaMemcpyDeviceToHost);
	for (k = 0; k < m; k++)
	{
		b[k] = b[k] / a[k][k];
	}
	for (k = 0; k < m; k++)
	{
		if (b[k] < 10e-5 && b[k]>-10e-5)
		{
			b[k] = 0;
		}
	}
	printf("The result is:\n");
	for (k = 0; k < m; k++)
	{
		printf("x%d = %g\n", k + 1, b[k]);
	}
	alltime = finish - start;
	//Test the result
	/*matmultCUDA(A_dev, m, b, 1, GnUn, 1, m, m, 1);
	printf("Test the result GnUn=:\n");
	for (i = 0; i < m; i++)
	{
		printf("%f\n", GnUn[i]);
	}*/

	cudaFree(A_dev);
	cudaFree(b_dev);
	printf("time to do is %lf ms \n", alltime);
	getchar();
	return 0;
}
int main()
{
	if (!InitCUDA()) {
		return 0;
	}
	printf("\n");
	double start, end;
	//Define the matrix
	float* A, * Es, * Js, * Gb, * Gn, * Jn, * Un, * Ib, * AT, * AGb, * AGbEs, * AJs;
	float* GnUn;
	//matrix input by hand
	A = (float*)malloc(sizeof(float) * m * n);// A incidience matrix m*n
	Es = (float*)malloc(sizeof(float) * n * 1);// Es The branch voltage source column vector n*1
	Js = (float*)malloc(sizeof(float) * n * 1);// Js The branch current source column vector n*1
	Gb = (float*)malloc(sizeof(float) * n * n);// Gb Branch conductance matrix n*n diagonal matrix
	//A matrix used to construct an equation
	Gn = (float*)malloc(sizeof(float) * m * m);//Gn Node conductance matrix m*m
	Jn = (float*)malloc(sizeof(float) * m * 1);//Jn Node current source column vector m*1
	//The intermediate matrix
	AT = (float*)malloc(sizeof(float) * n * m);// A Transposed matrix n*m
	AGb = (float*)malloc(sizeof(float) * m * n);// A * Gb Mediation matrix
	AGbEs = (float*)malloc(sizeof(float) * m * 1);// A * Gb * Es Mediation matrix
	AJs = (float*)malloc(sizeof(float) * m * 1);// A * Js Mediation matrix
	//The result
	Un = (float*)malloc(sizeof(float) * m * 1);//Un Node voltage column vector m*1
	Ib = (float*)malloc(sizeof(float) * m * 1);//Ib Branch current column vector m*1
	//Test the result
	GnUn = (float*)malloc(sizeof(float) * m * 1);
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
	mat_cal_Gn(A, AT, Gb, AGb, Gn, m, n);
	//Begin to calculate Jn
	mat_cal_Jn(A, Es, AGb, AGbEs, Js, AJs, Jn, m, n);
	//Transform Gn and Jn into two-dimensional matrix
	int i, j;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < m; j++)
		{
			a[i][j] = Gn[i * m + j];
		}
	}
	for (i = 0; i < m; i++)
	{
		b[i] = Jn[i];
	}
	printf("\n");
	matrix_sloving();
	//End of the timing
	end = clock();
	printf("Time consumed %f ms\n", end - start);
	return 0;
}
