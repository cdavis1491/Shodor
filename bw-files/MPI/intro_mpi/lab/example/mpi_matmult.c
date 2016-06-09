/**************************************
 * Filename: mpi_matmult.c
 * Distription: matZ = MatX * matY
 * 
 * How to compile:
 *  $ make mpi_matmult
 * How to run on 4 cores:
 *  $ aprun -n 4 ./mpi_matmult.exe
 ***************************************/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

//Global variables
//double **matX, **matY, **matZ;
// Functions Declaration
// allocMem wrapper to allocate memory for each matrix
//void allocMem();
// freeMem wrapper to take back allocated memory
//void freeMem();
//print test 
//void printMat();

#define NUM_ROW 200   //Number of rows in each matrix
#define NUM_COL 200   //Number of column in each matrix
#define MASTER 0       //make process zero be the master

int main (int argc, char **argv){
    
    double **matX;
    double **matY;
    double **matZ;    //seting to null remove usage error
    
    matX = malloc(sizeof(double) * NUM_ROW * NUM_COL);
    matY = malloc(sizeof(double) * NUM_ROW * NUM_COL);
    matZ = malloc(sizeof(double) * NUM_ROW * NUM_COL);


    //allocMem();             //Allocating Memory
    
    int i,j,k;              //loop index variables
    
    int num_ps, my_rank;
    int numWorkers;
    int source, dist;
    int rows, tagMaster, tagWorker;
    double start_t, end_t, compute_t = 0.0;
    int matBSize;
    int evenSplit,remainder, offset;

    MPI_Status status;
    MPI_Init(&argc,&argv);
    
    //Start the timer for the comput intense loop
    //start_t = MPI_Wtime();

    //get the number of processes in a group
    MPI_Comm_size(MPI_COMM_WORLD,&num_ps);

    //get the process ID number
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    
    //Assign an even number of rows to each rank, 
    //except for the last rank
    numWorkers = num_ps-1;

    if (my_rank == MASTER){
        printf ("Compute matrix product Z = X * Y.\n" );
        printf("Number of worker tasks = %d\n",numWorkers);
        //Initialize matA basically each element is the sum of index i+j.
        for ( i = 0; i < NUM_ROW; i++ ){
            for ( j = 0; j < NUM_COL; j++ ){
                matX[i][j] = 1;
            }
        } //END: outerloop
        //Initialize matB basically to product of indicies for each element.
        for ( i = 0; i < NUM_ROW; i++ ){
            for ( j = 0; j < NUM_COL; j++ ){
                matY[i][j] = 2;
            }
        } //END: outerloop

    
    //Split the work among threads as even as possible
    evenSplit = NUM_ROW / numWorkers; //for even split
    remainder = NUM_ROW % numWorkers; 
    offset = 0;
    tagMaster = 7;
     
    for (dist=1; dist<=numWorkers; dist++){
        rows = (dist <= remainder) ? evenSplit+1 : evenSplit;
        printf("   sending %d rows to task %d\n",rows,dist);
        
        matBSize = rows*NUM_COL;
        //where to start in matX
        MPI_Send(&offset, 1, MPI_INT, dist, tagMaster, MPI_COMM_WORLD);
        
        //Send num rows each ps to compute 
        MPI_Send(&rows, 1, MPI_INT, dist, tagMaster, MPI_COMM_WORLD);
        MPI_Send(&matX[offset][0], matBSize, MPI_DOUBLE, 
                dist, tagMaster, MPI_COMM_WORLD);
        
        MPI_Send(&matY, matBSize, MPI_DOUBLE, dist, tagMaster, 
                MPI_COMM_WORLD);
        
        offset = offset + rows;
    } //END: first send

    
    //Workers start from here on!!
    tagWorker = 8;
    for (i=1; i<=numWorkers; i++){
        source = i;
        //For every send there should be a MPI_Recv() call
        //Get the offset value send by master
        MPI_Recv(&offset, 1, MPI_INT, source, tagWorker, 
                MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, tagWorker, 
                MPI_COMM_WORLD, &status);
        MPI_Recv(&matZ[offset][0], matBSize, MPI_DOUBLE, source, 
                tagWorker,MPI_COMM_WORLD, &status);

    } //END: first Receive Loop
        //Print 10 rows and 6 columns for testing value.
    //printMat();
    
    }//END: master initialization of matrix

    //----rest of the threads, started messaging---
    if (my_rank > MASTER){
        tagMaster = 7;
        //Get initial offset value
        MPI_Recv(&offset, 1, MPI_INT, 0, tagMaster, 
                MPI_COMM_WORLD, &status);
        //get rows to compute
        MPI_Recv(&rows, 1, MPI_INT, 0, tagMaster, 
                MPI_COMM_WORLD, &status);
        //get matX
        MPI_Recv(&matX, matBSize, MPI_DOUBLE, 0, tagMaster, 
                MPI_COMM_WORLD, &status);
        //receive the matY
        MPI_Recv(&matY, matBSize, MPI_DOUBLE, 0, tagMaster, 
                MPI_COMM_WORLD, &status);
        // Compute matZ = matX * matY.
        for ( i = 0; i < NUM_ROW; i++ ){
            for ( j = 0; j < NUM_COL; j++ ){
                matZ[i][j] = 0.0;
                for ( k = 0; k < NUM_ROW; k++ ){
                    //Actuall multiplication here.
                    matZ[i][j] = matZ[i][j] + matX[i][k] * matY[k][j];
                }
            }
        } //END: outerloop
        tagWorker = 8;
        //send back to master the computed portion
        MPI_Send(&offset, 1, MPI_INT, MASTER,tagWorker,
                MPI_COMM_WORLD);
        //send back num rows worked on
        MPI_Send(&rows, 1, MPI_INT, MASTER, tagWorker, 
                MPI_COMM_WORLD);
        //send resulting matrix
        MPI_Send(&matZ, matBSize, MPI_DOUBLE, MASTER,
                tagWorker, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    
    //how long program spent in last loop
    //end_t = MPI_Wtime();
    //compute_t = end_t - start_t;
    printf (" Compute time in seconds = %g \n", compute_t );
     
    //Call function to free memory allocated for each matrix
    //freeMem();
    return 0;
} //END: main()

/*Allocate Memory to each matrix
void allocMem(){
    int i;
    matX = (double **)malloc(NUM_ROW*sizeof(double *));
    for (i=0;i<NUM_ROW; i++){
        matX[i]=(double *)malloc(NUM_COL*sizeof(double));
    }
    matY = (double **)malloc(NUM_ROW*sizeof(double *));
    for (i=0;i<NUM_ROW; i++){
        matY[i]=(double *)malloc(NUM_COL*sizeof(double));
    }
    matZ = (double **)malloc(NUM_ROW*sizeof(double *));
    for (i=0;i<NUM_ROW; i++){
        matZ[i]=(double *)malloc(NUM_COL*sizeof(double));
    }
    //return 0;

} //END: allocMem() 

void printMat(){
    int i, j;
    printf("Computed first 6 rows are:\n");
    //printf("matSum(i,j)  = %8.5f\n", matSum[10][10] );
    for(i=0; i < 10; i++){
        printf(" \n");
        for(j=0; j< 6; j++){
            printf("  %.1f ", matZ[i][j] );
        }
    }
    printf(" \n");
}
//Free allocated memory to two dimentional arrays
void freeMem(){
    int i;
    for (i=NUM_ROW-1; i>=0; i--){
        free(matX[i]);
    }
    free(matX);
    for (i=NUM_ROW-1; i>=0; i--){
        free(matY[i]);
    }
    free(matY);
    for (i=NUM_ROW-1; i>=0; i--){
        free(matZ[i]);
    }
    free(matZ);
    //return 0;
} //END: freeMem()


*/
