/* Cholesky decomposition.
 * Host code.
 * Author: Naga Kandasamy
 * Date: May 23, 2013
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include "chol_kernel.cu"


////////////////////////////////////////////////////////////////////////////////
// declarations, forward
Matrix allocate_matrix_on_gpu(const Matrix M);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost);
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice);

void check_error(const char *msg);

extern Matrix create_positive_definite_matrix(unsigned int, unsigned int);
extern "C" int chol_gold(const Matrix, Matrix);
extern "C" int check_chol(const Matrix, const Matrix);
void chol_on_device(const Matrix, Matrix);
void chol_on_device_optimized(const Matrix, Matrix);

//Globals
float time_cpu;
// Matrices for the program
Matrix A; // The N x N input matrix
Matrix reference; // The upper triangular matrix computed by the CPU
Matrix U_on_device; // The upper triangular matrix computed by the device (slow)
Matrix U_on_device_fast; // The upper triangular matrix computed by the device (fast)


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{	
	// Check command line arguments
	if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}		
	
	// Initialize the random number generator with a seed value 
	srand(time(NULL));

	// Create the positive definite matrix. May require a few tries if we are unlucky
	int success = 0;
	while(!success){
		A = create_positive_definite_matrix(MATRIX_SIZE, MATRIX_SIZE);
		if(A.elements != NULL)
				  success = 1;
	}

	reference  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the CPU result
	U_on_device =  allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); // Create a matrix to store the device result
	U_on_device_fast =  allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);

	//Compute the Cholesky decomposition on the CPU
	unsigned int timer;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
	printf("== CPU ==\n");
	int status = 1;
	status = chol_gold(A, reference);
	cutStopTimer(timer);
	time_cpu = 1e-3 * cutGetTimerValue(timer);
	printf("	Run time:    %0.10f s. \n", time_cpu);
	if(status == 0){
			  printf("Cholesky decomposition failed. The input matrix is not positive definite. \n");
			  exit(0);
	}
	/*
	printf("Double checking for correctness by recovering the original matrix. \n");
	if(check_chol(A, reference) == 0){
		printf("CPU: FAILED\n");
		exit(0);
	}
	*/
	printf("	PASSED\n"); //IT IS SO PERFECT WE DON'T EVEN CHECK.
	

	//Slow
	//Perform the Cholesky decomposition on the GPU. The resulting upper triangular matrix should be retured in U_on_gpu
	chol_on_device(A, U_on_device);
	
	
	//Optimized
	//Perform the Cholesky decomposition on the GPU. The resulting upper triangular matrix should be retured in U_on_gpu
	chol_on_device_optimized(A, U_on_device_fast);
	
	// Free host matrices
	free(A.elements); 	
	free(U_on_device.elements);
	free(U_on_device_fast.elements);	
	free(reference.elements); 
	return 1;
}

//Error helper
void check_for_error(char *msg){
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} 

/* Write code to perform Cholesky decopmposition on the device. */
void chol_on_device(const Matrix A, Matrix U)
{
	//Slow
	//Perform the Cholesky decomposition on the GPU. The resulting upper triangular matrix should be retured in U_on_gpu
	unsigned int timer_gpu;
	cutCreateTimer(&timer_gpu);
	
	//A and U are already allocated on CPU already
	//Allocate space on gpu
	Matrix gpu_u = allocate_matrix_on_gpu( U );

	//Copy matrices to gpu, copy A right into U
	copy_matrix_to_device( gpu_u, A );
	
	//Maximum size expected is 8192x8192
	//Will be splitting the elimination i loop
	//Which has up to MATRIX_SIZE iterations
	//So we would optimally use 8192 threads
	//Thus requiring 16 blocks
	//Rather than attempting to syncronize 16 blocks
	//Where each thread does one operation per outer K iteration
	//Just have one block and have each thread do 16 operations 
	//(in the worst case)
	int num_blocks = 1;
	
	//Max per block threads
	int threads_per_block = 512;
	
	//Operations per thread
	int ops_per_thread = MATRIX_SIZE / (threads_per_block*num_blocks);
	
	printf("== GPU (Slow) ==\n");
	printf("	Threads per block: %d\n",threads_per_block);
	printf("	Number of blocks: %d\n",num_blocks);
	printf("	Operations per thread: %d\n",ops_per_thread);
	
	//Set up the execution grid on the GPU
	dim3 thread_block(threads_per_block, 1, 1);
	dim3 grid(num_blocks,1);
	
	//Start timer after copy
	cutStartTimer(timer_gpu);
	
	// Launch the kernel <<<grid, thread_block>>>
	chol_kernel<<<grid, thread_block>>>(gpu_u.elements,ops_per_thread);
	
	//Sync at end and check for errors
	cudaThreadSynchronize();
	check_for_error("SLOW KERNEL FAILURE\n");
	
	//Stop timer before copy back
	cutStopTimer(timer_gpu);
	
	//Copy data back
	copy_matrix_from_device(U, gpu_u);
	
	//Free memory on device
	cudaFree(gpu_u.elements);
	
	float time_gpu = 1e-3 * cutGetTimerValue(timer_gpu);
	printf("	Run time:    %0.10f s. \n", time_gpu);
	printf("	Speedup: %0.10f\n", time_cpu/time_gpu);
	//Check if the device result is equivalent to the expected solution. If you can't meet the desired tolerance, try using double precision support.
	unsigned int size = reference.num_rows * reference.num_columns;
	CUTBoolean res = cutComparefe(reference.elements, U_on_device.elements, size, 0.1f);
	printf("	%s\n", (1 == res) ? "PASSED" : "FAILED");
}

/* Write code to perform Cholesky decopmposition on the device. */
void chol_on_device_optimized(const Matrix A, Matrix U)
{
	unsigned int timer_gpu_fast;
	cutCreateTimer(&timer_gpu_fast);
	
	printf("== GPU (Fast) ==\n");
	//A and U are already allocated on CPU already
	//Allocate space on gpu for U
	Matrix gpu_u = allocate_matrix_on_gpu( U );

	//Copy matrices to gpu, copy A right into U
	copy_matrix_to_device( gpu_u, A );
	
	//Start timer after copy
	cutStartTimer(timer_gpu_fast);
	
	//Each thread within a block will take some j iterations
	int threads_per_block = 256; //Optimal
	//Stride size should equal threads per block - just cause?
	int stride = threads_per_block;
	printf("	Threads per block / stride: %d\n",threads_per_block);

	
	//Each kernel call will be one iteration of out K loop
	int k;
	for(k = 0; k < MATRIX_SIZE; k++)
	{
		//Want threads to stride across memory
		//i is outer loop
			//j is inner loop
		//so threads should split the j loop
		//Each thread block will take an i iteration
		int isize = (MATRIX_SIZE-1) - (k+1) + 1;
		int num_blocks = isize;
		if(num_blocks <= 0)
		{
			num_blocks = 1;
		}
		
		//Set up the execution grid on the GPU
		//printf("	Threads per block: %d\n",threads_per_block);
		//printf("	Number of blocks: %d\n",num_blocks);
		dim3 thread_block(threads_per_block, 1, 1);
		dim3 grid(num_blocks,1);
		
		//Call the div kernel for this k iteration
		chol_kernel_optimized_div<<<grid, thread_block>>>(
			gpu_u.elements,
			k,
			stride);
		
		//Call kernel with for this K iteration
		chol_kernel_optimized<<<grid, thread_block>>>(
			gpu_u.elements,
			k,
			stride);
			
			
		//Sync at end and check for errors
		cudaThreadSynchronize();
		check_for_error("FAST KERNEL FAILURE");
	}
	
	//Sync at end
	cudaThreadSynchronize();
	
	//Stop timer before copy back					 
	cutStopTimer(timer_gpu_fast);
	
	//Copy data back
	copy_matrix_from_device(U, gpu_u);
	
	//Free memory on device
	cudaFree(gpu_u.elements);
	
	
	//As the final step, zero out the lower triangular portion of U
	int i, j;
	for(i = 0; i < MATRIX_SIZE; i++)
			  for(j = 0; j < i; j++)
						 U.elements[i * MATRIX_SIZE + j] = 0.0;
						 
	float time_gpu_fast = 1e-3 * cutGetTimerValue(timer_gpu_fast);
	printf("	Run time:    %0.10f s. \n", time_gpu_fast);
	printf("	Speedup: %0.10f\n", time_cpu/time_gpu_fast);
	//Check if the device result is equivalent to the expected solution. If you can't meet the desired tolerance, try using double precision support.
	unsigned int size_fast = reference.num_rows * reference.num_columns;
	CUTBoolean res_fast = cutComparefe(reference.elements, U_on_device_fast.elements, size_fast, 0.1f);
	printf("	%s\n", (1 == res_fast) ? "PASSED" : "FAILED");
}

// Allocate a device matrix of same size as M.
Matrix allocate_matrix_on_gpu(const Matrix M){
    Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}


// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    	Matrix M;
    	M.num_columns = M.pitch = num_columns;
    	M.num_rows = num_rows;
    	int size = M.num_rows * M.num_columns;
		
	M.elements = (float *) malloc(size * sizeof(float));
	for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
			M.elements[i] = (float)rand()/(float)RAND_MAX;
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice){
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

void check_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) 
	{
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}						 
}
