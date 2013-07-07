/* Reference code for the Cholesky decomposition. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "chol.h"

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" int chol_gold(const Matrix, Matrix);
extern "C" int check_chol(const Matrix, const Matrix);
Matrix matrix_multiply(const Matrix, const Matrix);
extern void print_matrix(const Matrix);
extern Matrix create_positive_definite_matrix(unsigned int, unsigned int);
extern void print_matrix(const Matrix);

/* Prints the matrix out to screen. */
void print_matrix(const Matrix M){
	for(unsigned int i = 0; i < M.num_rows; i++){
		for(unsigned int j = 0; j < M.num_columns; j++)
			printf("%f ", M.elements[i*M.num_rows + j]);
		printf("\n");
	} 
	printf("\n");
}


/* Function checks if the matrix is symmetric. */
int check_if_symmetric(const Matrix M)
{
	for(unsigned int i = 0; i < M.num_rows; i++)
		for(unsigned int j = 0; j < M.num_columns; j++)
			if(M.elements[i * M.num_rows + j] != M.elements[j * M.num_columns + i])
				return 0;
	return 1;
}

/* Function checks if the matrix is diagonally dominant. */
int check_if_diagonal_dominant(const Matrix M)
{
	float diag_element;
	float sum;
	for(unsigned int i = 0; i < M.num_rows; i++){
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for(unsigned int j = 0; j < M.num_columns; j++){
			if(i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		if(diag_element <= sum)
			return 0;
	}

	return 1;
}


/* A matrix M is positive definite if x^TMx > 0 for all non-zero vectors X. 
   A matrix M is positive definite if and only if the determinant of each of the principal submatrices is positive. 
   A diagonally dominant NxN symmetric matrix is positive definite. This function generates a diagonally dominant NxN symmetric matrix. */
Matrix create_positive_definite_matrix(unsigned int num_rows, unsigned int num_columns)
{
	Matrix M;
	M.num_columns = M.pitch = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));

	// Step 1: Create a matrix with random numbers between [-.5 and .5]
	printf("Creating a %d x %d matrix with random numbers between [-.5, .5]...", num_rows, num_columns);
	unsigned int i;
	unsigned int j;
	for(i = 0; i < size; i++)
		M.elements[i] = ((float)rand()/(float)RAND_MAX) - 0.5;
       	printf("done. \n");
	// print_matrix(M);
	// getchar();

	// Step 2: Make the matrix symmetric by adding its transpose to itself
	printf("Generating the symmetric matrix...");
	Matrix transpose;
	transpose.num_columns = transpose.pitch = num_columns;
	transpose.num_rows = num_rows; 
	size = transpose.num_rows * transpose.num_columns;
	transpose.elements = (float *)malloc(size * sizeof(float));

	for(i = 0; i < M.num_rows; i++)
		for(j = 0; j < M.num_columns; j++)
			transpose.elements[i * M.num_rows + j] = M.elements[j * M.num_columns + i];
	// print_matrix(transpose);

	for(i = 0; i < size; i++)
		M.elements[i] += transpose.elements[i];
	if(check_if_symmetric(M))
		printf("done. \n");
	else{ 
		printf("error. \n");
		free(M.elements);
		M.elements = NULL;
	}
	// print_matrix(M);
	// getchar();

	// Step 3: Make the diagonal entries large with respect to the row and column entries
	printf("Generating the positive definite matrix...");
	for(i = 0; i < num_rows; i++)
		for(j = 0; j < num_columns; j++){
			if(i == j) 
				M.elements[i * M.num_rows + j] += 0.5 * M.num_rows;
		}
	if(check_if_diagonal_dominant(M))
		printf("done. \n");
	else{
		printf("error. \n");
		free(M.elements);
		M.elements = NULL;
	}
	// print_matrix(M);
	// getchar();

	free(transpose.elements);

	// M is diagonally dominant and symmetric!
	return M;
}


/* This function implements a row-oriented Cholesky decomposition on the input matrix A to generate an upper triangular matrix U 
	such that A = U^TU. 
 */	
int chol_gold(const Matrix A, Matrix U)
{
	unsigned int i, j, k; 
	unsigned int size = A.num_rows * A.num_columns;

	// Copy the contents of the A matrix into the working matrix U
	for (i = 0; i < size; i ++)
		U.elements[i] = A.elements[i];

	// Perform the Cholesky decomposition in place on the U matrix
	for(k = 0; k < U.num_rows; k++){
			  // Take the square root of the diagonal element
			  U.elements[k * U.num_rows + k] = sqrt(U.elements[k * U.num_rows + k]);
			  if(U.elements[k * U.num_rows + k] <= 0){
						 printf("Cholesky decomposition failed. \n");
						 return 0;
			  }
			  
			  // Division step
			  for(j = (k + 1); j < U.num_rows; j++)
						 U.elements[k * U.num_rows + j] /= U.elements[k * U.num_rows + k]; // Division step
						 

			  // Elimination step
			  for(i = (k + 1); i < U.num_rows; i++)
						 for(j = i; j < U.num_rows; j++)
									U.elements[i * U.num_rows + j] -= U.elements[k * U.num_rows + i] * U.elements[k * U.num_rows + j];

	}

	// As the final step, zero out the lower triangular portion of U
	for(i = 0; i < U.num_rows; i++)
			  for(j = 0; j < i; j++)
						 U.elements[i * U.num_rows + j] = 0.0;

	// printf("The Upper triangular matrix is: \n");
	// print_matrix(U);

	return 1;
}

/* Helper function which checks the correctness of Cholesky decomposition by attempting to recover the original matrix A using U^TU. */
int check_chol(const Matrix A, const Matrix U)
{
	Matrix U_transpose;
	U_transpose.num_columns = U_transpose.pitch = U.num_columns;
	U_transpose.num_rows = U.num_rows; 
	unsigned int size = U_transpose.num_rows * U_transpose.num_columns;
	U_transpose.elements = (float *)malloc(size * sizeof(float));

	// Determine the transpose of U
	unsigned int i, j;
	for(i = 0; i < U.num_rows; i++)
		for(j = 0; j < U.num_columns; j++)
			U_transpose.elements[i * U.num_rows + j] = U.elements[j * U.num_columns + i];

	// Multiply U and U_transpose
	Matrix A_recovered = matrix_multiply(U_transpose, U);
	// print_matrix(A_recovered);

	// Compare the two matrices A and A_recovered
	for(i = 0; i < size; i++)
			  if(fabs(A.elements[i] - A_recovered.elements[i]) > 0.01)
						 return 0;

	return 1;

}

/* Helper function that multiplies two given matrices and returns the result. */
Matrix matrix_multiply(const Matrix A, const Matrix B)
{
		  Matrix C;
		  C.num_columns = C.pitch = A.num_columns;
		  C.num_rows = A.num_rows; 
		  unsigned int size = C.num_rows * C.num_columns;
		  C.elements = (float *)malloc(size * sizeof(float));

		  for (unsigned int i = 0; i < A.num_columns; i++)
					 for (unsigned int j = 0; j < B.num_rows; j++){
								double sum = 0.0f;
								for (unsigned int k = 0; k < A.num_columns; k++){
										  double a = A.elements[i * A.num_columns + k];
										  double b = B.elements[k * B.num_rows + j];
										  sum += a * b;
								}
								C.elements[i * B.num_rows + j] = (float)sum;
					 }
		  return C;
}
	
