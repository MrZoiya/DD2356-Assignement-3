
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define SEED     921

int main(int argc, char* argv[])
{
    int local_count = 0, count = 0, num_iter = 1000000000;
    int rank, num_ranks, provided, num_iter_local;
    double x, y, z, pi;
    
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

    double start_time, stop_time, elapsed_time;
    start_time = MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    
    printf("Hello from process %d\n",rank);

    srand(SEED*rank); // Important: Multiply SEED by "rank" when you introduce MPI!
    
    num_iter_local = num_iter/num_ranks;

    // Calculate PI following a Monte Carlo method
    for (int iter = 0; iter < num_iter_local; iter++)
    {
        // Generate random (X,Y) points
        x = (double)random() / (double)RAND_MAX;
        y = (double)random() / (double)RAND_MAX;
        z = sqrt((x*x) + (y*y));
        
        // Check if point is in unit circle
        if (z <= 1.0)
        {
            local_count++;
        }
    }
    
    // Sum up the intermediate counts on rank 0
    if (rank != 0)
    {
        MPI_Send(&local_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        count = local_count;
        for (int i = 1; i < num_ranks; i++)
        {
            MPI_Recv(&local_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            count += local_count;
        }
    }

    stop_time = MPI_Wtime();
  
    // Calculate PI estimate on rank 0 and display the result
    if (rank == 0)
    {
        // Estimate Pi and display the result
        pi = ((double)count / (double)num_iter) * 4.0;
        printf("The result is %f\n", pi);

        // Calculate the elapsed time
        elapsed_time = stop_time - start_time;
        printf("Elapsed time: %f seconds\n", elapsed_time);
    }
        
    MPI_Finalize();
    return 0;
}

