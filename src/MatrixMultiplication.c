/*
 ================================================================================================================
 Name        : MatrixMultiplication.c
 Author      : Roberto Gagliardi
 Description : Matrix multiplication using MPI
 ================================================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

// Utility functions prototypes
void initMatrix(int **matrix, int size, int mod);
void allocMatrix(int **matrix, int rows, int columns);
void printMatrix(int **matrix, int rows, int columns, char *name);

int main(int argc, char* argv[]) {

	  // Initialize MPI environment
	  MPI_Init(&argc, &argv);

	  // Get number of processes
	  int world_size;
	  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	  // Get rank of process
	  int rank;
	  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	  // Get name of processor
	  char processor_name[MPI_MAX_PROCESSOR_NAME];
	  int name_len;
	  MPI_Get_processor_name(processor_name, &name_len);

	  // Hello message
	  printf("Hello from processor %s, rank %d out of %d processors\n", processor_name, rank, world_size);

	  // Dimensions
	  int size = atoi(argv[1]);			// Size of both rows and columns of matrices
	  int s = size / world_size;		// Subset of rows and columns per processor

	  // Check size of matrices
	  if (size % world_size != 0) {
		  if (rank == 0)
			  printf("Size of matrices must be divisible by the number of processors\n");
		  MPI_Finalize();
		  return 0;
	  }

	  // Matrices allocation
	  int **matrixA, **matrixB, **matrixC;
	  matrixA = (int **) malloc(size * sizeof(int *));
	  allocMatrix(matrixA, size, size);
	  matrixB = (int **) malloc(size * sizeof(int *));
	  allocMatrix(matrixB, size, size);
	  matrixC = (int **) malloc(size * sizeof(int *));
	  allocMatrix(matrixC, size, size);

	  // Master processor
	  if (rank == 0) {

		  // Matrices initialization
		  initMatrix(matrixA, size, world_size);
		  printMatrix(matrixA, size, size, "First matrix");
		  initMatrix(matrixB, size, world_size);
		  printMatrix(matrixB, size, size, "Second matrix");

	  }

	  // Time of start processing
	  double startTime = MPI_Wtime();

	  // Send subset of first matrix to each other processor
	  MPI_Scatter(matrixA[rank * s], s * size, MPI_INT, matrixA[rank * s], s * size, MPI_INT, 0, MPI_COMM_WORLD);

	  // Send second matrix to each other processor
	  MPI_Bcast(matrixB[0], size * size, MPI_INT, 0, MPI_COMM_WORLD);

	  // Algorithm for matrix multiplication
	  for (int i = rank * s; i < (rank + 1) * s; i++) {
		  for (int j = 0; j < size; j++) {
			  matrixC[i][j] = 0;
			  for (int k = 0; k < size; k++) {
				  matrixC[i][j] = matrixC[i][j] + matrixA[i][k] * matrixB[k][j];
			  }
		  }
	  }

	  // Receive result matrix by each other processor
	  MPI_Gather(matrixC[rank * s], s * size, MPI_INT, matrixC[rank * s], s * size, MPI_INT, 0, MPI_COMM_WORLD);

	  // Time of end processing
	  double endTime = MPI_Wtime();

	  // Master processor
	  if (rank == 0) {

		  // Print result matrix
		  printMatrix(matrixC, size, size, "Result matrix");

		  // Print time spent
		  printf("The operation took %f seconds\n", endTime - startTime);

	  }

	  // Free allocated memory
	  free(matrixA);
	  free(matrixB);
	  free(matrixC);

	  // Finalize the MPI environment
	  MPI_Finalize();
	  return 0;

}

// Allocation of matrix as array of successive elements
void allocMatrix(int **matrix, int rows, int columns) {
	int *elements = (int *) malloc(rows * columns * sizeof(int));
	for (int i = 0; i < rows; i++)
		matrix[i] = &elements[i * columns];
}

// Initialize matrix with successive values between 1 and 'mod'
void initMatrix(int **matrix, int size, int mod) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			matrix[i][j] = (i + j) % mod + 1;
		}
	}
}

// Print matrix with the specified name, rows and columns
void printMatrix(int **matrix, int rows, int columns, char *name) {
	printf("%s:\n", name);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			printf("%d, ", matrix[i][j]);
		}
		printf("\n");
	}
}
