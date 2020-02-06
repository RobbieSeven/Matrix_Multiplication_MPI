# Parallel Matrix Multiplication

### Corso di Programmazione Concorrente, Parallela e su Cloud

### Università degli Studi di Salerno - Anno Accademico 2019/20

**Prof. Vittorio Scarano**

**Dott. Carmine Spagnuolo**

**Roberto Gagliardi 0522500543**

## Problema

La moltiplicazione tra una matrice A di dimensioni *m x n* e una matrice B di dimensioni *n x l* risulta in un una matrice C di dimensioni *m x l*, così come viene raffigurato di seguito.

![image](img/Matrix_multiplication.png)

## Soluzione

Il programma C esposto nel file *src/matrixMultiplication.c* fornisce una soluzione al problema della moltiplicazione tra due matrici quadrate di dimensioni *N x N*.

Tale soluzione si serve del calcolo parallelo, suddividendo il carico di lavoro su più processori utilizzando delle operazioni di comunicazione collettiva fornite da MPI, come le funzioni di *broadcast*, *scatter* e *gather*.

Il calcolo del prodotto avviene assumendo che le dimensioni delle matrici siano divisibili per il numero di processori coinvolti nell'operazione, così da poter garantire l'assegnamento dello stesso carico di lavoro a ciascun processore.

In particolare, la soluzione prevede che ogni processore abbia il compito di calcolare una porzione della matrice risultante C. Per fare questo, il processore master, dopo aver inizializzato le due matrici da moltiplicare, invierà ad ogni altro processore un sottoinsieme della matrice A e l'intera matrice B, così come illustrato dalla figura sottostante.

![image](img/Matrix_breakdown.png)

In questo modo, tutti i processori (compreso il master) potranno calcolare la propria parte della soluzione, che consiste in un sottoinsieme delle righe della matrice C. Fatto questo, ogni processore invierà tale risultato al master, che avrà il compito di assemblare tutte le porzioni del risultato in un'unica matrice finale.

Nel caso in cui le matrici fornite non siano quadrate o che le loro dimensioni non siano divisibili per il numero di processori stabilito, il programma si arresterà e non procederà al calcolo del risultato.

## Implementazione

### Inizializzazione

La prima parte della soluzione prevede tutte le opportune inizializzazioni dell'ambiente MPI, così da recuperare il numero di processori e il rango di quello corrente.

```c
// Initialize MPI environment
MPI_Init(&argc, &argv);

// Get number of processes
int world_size;
MPI_Comm_size(MPI_COMM_WORLD, &world_size);

// Get rank of process
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
```

Successivamente, si passa a recuperare la dimensione delle matrici e a verificare che essa sia divisibile per il numero di processori. Oltre a questo, si procede a determinare anche il numero di righe della matrice A che, in seguito, saranno assegnate a ciascun processore. Tale valore si ottiene semplicemente dividendo la dimensione delle matrici per il numero di processori.

```c
// Dimensions
int size = atoi(argv[1]);		// Size of both rows and columns of matrices
int s = size / world_size;	    // Subset of rows and columns per processor

// Check size of matrices
if (size % world_size != 0) {
    if (rank == 0)
        printf("Size of matrices must be divisible by the number of processors");
    MPI_Finalize();
    return 0;
}
```

Dopodiché, il programma si occupa di allocare la memoria necessaria da destinare alle matrici.

```c
// Matrices allocation
int **matrixA, **matrixB, **matrixC;
matrixA = (int **) malloc(size * sizeof(int *));
allocMatrix(matrixA, size, size);
matrixB = (int **) malloc(size * sizeof(int *));
allocMatrix(matrixB, size, size);
matrixC = (int **) malloc(size * sizeof(int *));
allocMatrix(matrixC, size, size);
```

La procedura seguente fa in modo che le matrici vengano allocate come un array di puntatori a blocchi contigui di memoria, riga per riga.

```c
// Allocation of matrix as an array of successive elements
void allocMatrix(int **matrix, int rows, int columns) {
	int *elements = (int *) malloc(rows * columns * sizeof(int));
	for (int i = 0; i < rows; i++)
		matrix[i] = &elements[i * columns];
}
```

Una volta allocata la memoria in maniera opportuna, il processore master (di rango 0) può inizializzare i valori delle matrici.

```c
// Master processor
if (rank == 0) {

    // Matrices initialization
    initMatrix(matrixA, size, world_size);
    printMatrix(matrixA, size, size, "First matrix");
    initMatrix(matrixB, size, world_size);
    printMatrix(matrixB, size, size, "Second matrix");

}
```

In questo caso, la funzione di inizializzazione non fa altro che assegnare ad ogni matrice dei valori compresi tra 1 e il numero totale di processori. Con la funzione di stampa, invece, si provvede a mostrare a video le due matrici che saranno moltiplicate.

 ```c
// Initialize matrix with successive values between one and 'mod'
void initMatrix(int **matrix, int size, int mod) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			matrix[i][j] = (i + j) % mod + 1;
		}
	}
}

// Print matrix with the specified name, where 'size' is the number of rows and columns
void printMatrix(int **matrix, int rows, int columns, char *name) {
	printf("%s:\n", name);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			printf("%d, ", matrix[i][j]);
		}
		printf("\n");
	}
}
```

### Calcolo della soluzione

A questo punto, il programma procede al calcolo della moltiplicazione tra le due matrici.

**Invio della prima matrice**

La prima cosa da fare è inviare a tutti i processori una porzione della matrice A. Per fare questo, la funzione di *scatter* mostrata di seguito fa in modo che il processore di rango 0, ovvero il master, invii a tutti gli altri processori un sottoinsieme delle righe della matrice A.

```c
// Send subset of first matrix to each other processor
MPI_Scatter(matrixA[rank * s], s * size, MPI_INT, matrixA[rank * s], s * size, MPI_INT, 0, MPI_COMM_WORLD);
```

Ricordando che:

- *rank* è il rango del processo corrente,

- *size* è la dimensione delle matrici e

- *s* è il numero di righe per ciascun processore,

la porzione della matrice da inviare viene determinata da:

- L'indice di partenza *rank * s*.

- Il numero di elementi da inviare *s * size*.

In questo modo, si garantisce che a ogni processore venga inviato lo stesso numero di righe *s* e tutte le colonne della matrice A, così come'era stato stabilito in precedenza. 

**Invio della seconda matrice**

La funzione di *broadcast* seguente, invece, fa in modo che il master invii a tutti gli altri processori la matrice B. In questo caso, la matrice viene inviata per intero, così da permettere a tutti di calcolare la propria parte della soluzione.

```c
// Send second matrix to each other processor
MPI_Bcast(matrixB[0], size * size, MPI_INT, 0, MPI_COMM_WORLD);
```

**Moltiplicazione delle matrici**

L'algoritmo seguente calcola la moltiplicazione tra il sottoinsieme della matrice A e la matrice B. Il risultato di questa operazione viene memorizzato nella matrice C.

```c
// Algorithm for matrix multiplication
int c = 0;
for (int i = rank * s; i < (rank + 1) * s; i++) {
    for (int j = 0; j < size; j++) {
        matrixC[i][j] = 0;
        for (int k = 0; k < size; k++) {
            matrixC[i][j] = matrixC[i][j] + matrixA[i][k] * matrixB[k][j];
            c++;
        }
    }
}
```

Da notare come anche il processore master, il cui rango è pari a 0, proceda a calcolare la propria porzione della soluzione, partendo dalla posizione *[0, 0]* della matrice. L'immagine seguente illustra le porzioni della matrice risultante C calcolate da ciascun processore.

![image](img/Matrix_result.png)

**Invio della soluzione**

A questo punto, ogni processore è pronto ad inviare la propria parte del risultato al processore master. Per fare questo, la seguente funzione di *gather* raccoglie nella matrice C del master tutte le porzioni calcolate da ogni altro processore.

```c
// Receive result matrix by each other processor
MPI_Gather(matrixC[rank * s], s * size, MPI_INT, matrixC[rank * s], s * size, MPI_INT,
            0, MPI_COMM_WORLD);
```

### Finalizzazione

Una volta concluso il calcolo della moltiplicazione tra le matrici, ogni processore procede a liberare la memoria allocata e a chiudere il proprio ambiente di MPI. Prima di fare ciò, il processore master si occupa di mostrare a video la matrice risultante.

```c
// Master processor
if (rank == 0) {

    // Print result matrix
    printMatrix(matrixC, size, size, "Result matrix");

}

// Free allocated memory
free(matrixA);
free(matrixB);
free(matrixC);

// Finalize the MPI environment
printf("Processor %d has done %d operations\n", rank, c);
MPI_Finalize();
return 0;
```
