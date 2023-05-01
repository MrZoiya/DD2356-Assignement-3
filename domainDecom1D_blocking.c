#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]){

    int rank, size, i, provided;
    
    // number of cells (global)
    int nxc = 128; // make sure nxc is divisible by size
    double L = 2*3.1415; // Length of the domain
    

    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // number of nodes (local to the process): 0 and nxn_loc-1 are ghost cells 
    int nxn_loc = nxc/size + 3; // number of nodes is number cells + 1; we add also 2 ghost cells
    double L_loc = L/((double) size);
    double dx = L / ((double) nxc);
    
    // define out function
    double *f = calloc(nxn_loc, sizeof(double)); // allocate and fill with z
    double *dfdx = calloc(nxn_loc, sizeof(double)); // allocate and fill with z

    for (i=1; i<(nxn_loc-1); i++)
      f[i] = sin(L_loc*rank + (i-1) * dx);
    
    // need to communicate and fill ghost cells f[0] and f[nxn_loc-1]
    // communicate ghost cells
    int previous = (rank == 0) ? size - 1 : rank - 1;
    int next = (rank == size - 1) ? 0 : rank+1;

    MPI_Send(&f[2], 1, MPI_DOUBLE, previous, 0, MPI_COMM_WORLD);
    MPI_Recv(&f[nxn_loc-1], 1, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(&f[nxn_loc-3], 1, MPI_DOUBLE, next, 1, MPI_COMM_WORLD);
    MPI_Recv(&f[0], 1, MPI_DOUBLE, previous, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // calculate first order derivative using central difference
    // here we need to correct value of the ghost cells!
    for (i=1; i<(nxn_loc-1); i++)
      dfdx[i] = (f[i+1] - f[i-1])/(2*dx);

    if (fabs(f[0] - sin(L_loc*rank - dx)) > 0.01 || fabs(f[nxn_loc-1] - sin(L_loc*(rank+1) + dx)) > 0.01 )
      printf("Incorrect ghost cell value, value computed : %f and %f, expected %f and %f\n",f[0], f[nxn_loc-1], sin(L_loc*rank - dx), sin(L_loc*(rank+1) + dx));
    else printf("Correct ghost shell for rank %d \n", rank);
    
    // Print f values
    if (rank==0){ // print only rank 0 for convenience
        if(fabs(1 -dfdx[1]) > 0.01)
            printf("Incorrect left ghost cell derivative, value computed : %f\n", dfdx[1]);
    }
    if(rank == size-1){
        if(fabs(1 -dfdx[nxn_loc-2]) > 0.01)
          printf("Incorrect right ghost cell derivative, value computed : %f\n", dfdx[nxn_loc-2]);
    }

    MPI_Finalize();
}






