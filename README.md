This is an implementation of a parallel algorithm using the Pthread C library for the resolution of linear systems using the gauss method.

**to compile**:

- mpicc -c GaussMPI.c
- mpicc -o GaussMPI GaussMPI.o

**to execute**:

- mpirun -np < número de processos> ./GaussMPI < tamanho da matriz> < seed> < arquivo de saida>
