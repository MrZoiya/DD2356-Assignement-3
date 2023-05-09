#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size, provided;
    
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hello World from rank %d from %d processes!\n", rank, size);
    
    MPI_Finalize();
    return 0;
}

