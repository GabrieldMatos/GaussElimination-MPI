/* Gaussian elimination without pivoting.
 * mpicc -c GaussMPI.c
 * mpicc -o GaussMPI GaussMPI.o
 * mpirun -np <número de processos> ./GaussMPI <tamanho da matriz> <seed> <arquivo de saida>
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>
#include <string.h>

#define MAXN 3000 /* Valor máximo para o tamanho da matriz NxN */
int N;

/* nome dado ao processo 0 */
#define MASTER 0

/* número de processos */
int totalDeProcessos;

int rankProcesso;

/* Nome do arquivo de saida*/
char *arquivoDeSaida;

/* Matriz do sistema, vetor B, e vetor de resposta X */
float A[MAXN][MAXN], B[MAXN], X[MAXN];

#define randm() 4 | 2 [uid] & 3

void gauss();

void backSubstitution();

/* srand baseado no tempo*/
unsigned int time_seed()
{
    struct timeval t;
    struct timezone tzdummy;

    gettimeofday(&t, &tzdummy);
    return (unsigned int)(t.tv_usec);
}

/* parametros do programa da linha de comando */
void parametros(int argc, char **argv)
{
    int seed = 0;
    srand(time_seed());

    if (argc == 4)
    {

        seed = atoi(argv[2]);
        srand(seed);

        if (rankProcesso == MASTER)
        {
            printf("\n Eliminação de Gauss\n");
            printf("\nRandom seed = %i\n", seed);
            int length = strlen(argv[3]);
            arquivoDeSaida = (char *)malloc(length + 1);
            arquivoDeSaida = argv[3];
            printf("\n Arquivo de saida = %s\n", arquivoDeSaida);
        }

        /* Tamanho da matriz */
        N = atoi(argv[1]);

        if (N < 1 || N > MAXN)
        {
            if (rankProcesso == MASTER)
            {
                printf("N = %i is out of range.\n", N);
            }
            exit(0);
        }
    }
    else
    {
        if (rankProcesso == MASTER)
        {
            printf("use: %s <dimensao_da_matriz> <random seed> <arquivo_de_saida>\n", argv[0]);
        }
        exit(0);
    }

    /* Escrever parametros */
    if (rankProcesso == MASTER)
    {
        printf("\nDimensão da matriz N = %i.\n", N);
    }
}

/* inicializar A and B and X */
void inicializaEntradas()
{
    int row, col;

    printf("\nIniciando...\n");

    for (col = 0; col < N; col++)
    {
        for (row = 0; row < N; row++)
        {
            A[row][col] = (float)rand() / 32768.0;
        }
        B[col] = (float)rand() / 32768.0;
        X[col] = 0.0;
    }
}

/* Escrever matrizes de entrada */
void escreve_entradas(FILE *filePtr)
{
    int row, col;

    if (N < 10)
    {
        fprintf(filePtr, "\nA =\n\t");
        for (row = 0; row < N; row++)
        {
            for (col = 0; col < N; col++)
            {
                fprintf(filePtr, "%9.6f%s", A[row][col], (col < N - 1) ? ", " : ";\n\t");
            }
        }
        fprintf(filePtr, "\nB = [");
        for (col = 0; col < N; col++)
        {
            fprintf(filePtr, "%9.6f%s", B[col], (col < N - 1) ? "; " : "]\n");
        }
    }
}

void print_X(FILE *filePtr)
{
    int row;

    if (N < 100)
    {
        fprintf(filePtr, "\nX = [");
        for (row = 0; row < N; row++)
        {
            fprintf(filePtr, "%9.6f%s", X[row], (row < N - 1) ? "; " : "]\n");
        }
    }
}

int main(int argc, char **argv)
{
    /* variáveis para calcular tempo */
    struct timeval etstart, etstop;
    struct timezone tzdummy;
    clock_t etstart2, etstop2;
    unsigned long long usecstart, usecstop;
    struct tms cputstart, cputstop;
    FILE *filePtr;

    // Inicializar MPI
    MPI_Init(&argc, &argv);

    // Número de processos
    MPI_Comm_size(MPI_COMM_WORLD, &totalDeProcessos);

    // rank do processo
    MPI_Comm_rank(MPI_COMM_WORLD, &rankProcesso);

    parametros(argc, argv);

    /* Arquivo para escrever os resultados */
    if (rankProcesso == MASTER)
    {
        filePtr = fopen(arquivoDeSaida, "w+");
        if (filePtr == NULL)
        {
            printf("ERROR");
        }

        inicializaEntradas();

        escreve_entradas(filePtr);
        printf("\nStart.\n");
        gettimeofday(&etstart, &tzdummy);
        etstart2 = times(&cputstart);
    }

    /* Eliminação de Gauss */
    gauss();

    if (rankProcesso == MASTER)
    {

        escreve_entradas(filePtr);

        gettimeofday(&etstop, &tzdummy);
        etstop2 = times(&cputstop);
        printf("\nStop.\n");
        usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
        usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

        /* mostrar resultado */
        print_X(filePtr);

        /* Mostrar resultados de tempo */
        fprintf(filePtr, "\nTempo gasto = %g ms.\n", (float)(usecstop - usecstart) / (float)1000);
        fprintf(filePtr, "--------------------------------------------\n");
        printf("\nResultado no arquivo %s\n", arquivoDeSaida);
    }

    MPI_Finalize();

    exit(0);
}

void gauss()
{

    int norm, row, col, multiplicador[N], rownum[N];

    /* Processo master (0) broadcast master para todos os processos*/
    MPI_Bcast(&A[0][0], MAXN * MAXN, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (row = 0; row < N; row++)
    {
        rownum[row] = row % totalDeProcessos;
    }

    for (norm = 0; norm < N; norm++)
    {
        /* O processador 0 transmite a linha 
        do loop externo que está sendo processada 
        para todos os outros processadores */
        MPI_Bcast(&A[norm][norm], N - norm, MPI_FLOAT, rownum[norm], MPI_COMM_WORLD);
        MPI_Bcast(&B[norm], 1, MPI_FLOAT, rownum[norm], MPI_COMM_WORLD);
        for (row = norm + 1; row < N; row++)
        {
            if (rownum[row] == rankProcesso)
            {
                multiplicador[row] = A[row][norm] / A[norm][norm];
            }
        }
        for (row = norm + 1; row < N; row++)
        {
            if (rownum[row] == rankProcesso)
            {
                for (col = 0; col < N; col++)
                {
                    A[row][col] = A[row][col] - (multiplicador[row] * A[norm][col]);
                }
                B[row] = B[row] - (multiplicador[row] * B[norm]);
            }
        }
    }

    backSubstitution();
}

void backSubstitution()
{
    int row, col;
    if (rankProcesso == MASTER)
    {
        for (row = N - 1; row >= 0; row--)
        {
            X[row] = B[row];
            for (col = N - 1; col > row; col--)
            {
                X[row] -= A[row][col] * X[col];
            }
            X[row] /= A[row][row];
        }
    }
}