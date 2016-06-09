#include "matmul.h"
#include "mpi.h"

#define FIRST_RANK 0

int main(int argc,char **argv) {
    int 
        rank, // Local rank ID
        num_ranks, // Total number of ranks
        c, // Command line switch from getopt
        *row_counts, // Per-rank count of rows to process
        *recv_counts, // Per-rank count of cells to process
        *displs; // Cumulative per-rank cell displacement in destination matrix
    unsigned int 
        i, // Loop index
        start_row, // Local row to start at
        stop_row, // Local row to end at
        row_stride, // Number of rows between start and stop row
        local_cells, // Number of cells between start and stop row
        dst_cells; // Total number of cells in destination matrix
    bool print = false; // Controls whether to print matrices to STDERR
    struct matrix 
        m1, // Local copy of matrix 1
        m2, // Local copy of matrix 2
        local_dst_m, // Local copy of destination matrix
        dst_m; // Final copy of destination matrix

    // Initiailize MPI
    MPI_Init(&argc,&argv);
    // num_ranks will be used later to represent the number of processes
    MPI_Comm_size(MPI_COMM_WORLD,&num_ranks);
    // The position of the current process within num_ranks
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

#ifdef DEBUG
    fprintf(stderr,"Rank %d/%d - Hello World\n",rank,num_ranks);
#endif

    // Set to 0 to catch input and runtime errors
    m1.rows = m1.cols = m2.rows = m2.cols = 0;

    while((c = getopt(argc,argv, "x:y:a:b:p")) != -1) {
        switch(c) {
            case 'x':
                m1.cols = atoi(optarg);
                break;
            case 'y':
                m1.rows = atoi(optarg);
                break;
            case 'a':
                m2.cols = atoi(optarg);
                break;
            case 'b':
                m2.rows = atoi(optarg);
                break;
            case 'p':
                print = true;
                break;
            case '?':
                usage();
                exit(EXIT_FAILURE);
        }
    }

    if(m1.rows == 0 || m1.cols == 0 || m2.rows == 0 || m2.cols == 0) {
        fprintf(stderr,"Supply row and column counts!\n");
        usage();
        exit(EXIT_FAILURE);
    }

    // The number of columns in the first matrix must match the number of
    // rows in the second matrix, due to the matrix multiplication
    // algorithm
    if(m1.cols != m2.rows) {
        fprintf(stderr,"Invalid matrix dimensions!\n");
        exit(EXIT_FAILURE);
    }

    m1.matrix       = safe_malloc_int(
            m1.rows*m1.cols,
            "Allocating first matrix"
            );
    m2.matrix       = safe_malloc_int(
            m2.rows*m2.cols,
            "Allocating second matrix"
            );

    row_counts     = safe_malloc_int(
            num_ranks,
            "Allocating row_counts array"
            );

    recv_counts     = safe_malloc_int(
            num_ranks,
            "Allocating recv_counts array"
            );

    displs          = safe_malloc_int(
            num_ranks,
            "Allocating displs array"
            );

    if(rank == FIRST_RANK) {
        // Each thread will get a separate random seed
        unsigned int *random_seeds = init_random_seeds();
        init_matrix(&m1,random_seeds);
        init_matrix(&m2,random_seeds);

        // Allocate full destination matrix on first rank
        // Will be populated via MPI_Gather
        dst_m.rows      = m1.rows;
        dst_m.cols      = m2.cols;
        dst_cells       = dst_m.cols*dst_m.rows;
        dst_m.matrix    = safe_malloc_int(
                dst_cells,
                "Allocating destination matrix"
                );

        if(print) {
            fprintf(stderr,"Matrix 1\n");
            print_matrix(&m1);
            fprintf(stderr,"\n");
            fprintf(stderr,"Matrix 2\n");
            print_matrix(&m2);
            fprintf(stderr,"\n");
        }

        // The displacement for rank 0 will always be 0
        displs[0] = 0;

        /* 
         * Each cell of row_counts will hold the number of rows the current
         * rank will process. Each rank except the last one will process the
         * same number of rows.
         */
        for(i=0;i<num_ranks-1;i++) {
            row_counts[i] = dst_m.rows / num_ranks;
#ifdef DEBUG
            fprintf(stderr,"Rank %d - row_counts[%d] is %d\n",rank,i,row_counts[i]);
#endif
        }
        // The last rank will process the base number of rows, plus the remaining
        // rows
        row_counts[num_ranks-1] = dst_m.rows / num_ranks
            + (dst_m.rows % num_ranks);
#ifdef DEBUG
        fprintf(stderr,"Rank %d - row_counts[%d] is %d\n",rank,(num_ranks-1),row_counts[num_ranks-1]);
#endif

        // recv_counts will hold the number of cells each rank will process
        for(i=0;i<num_ranks;i++) {
            recv_counts[i] = row_counts[i] * dst_m.cols;
        }
        // displs will hold the cell displacement for each rank, which
        // will be equial to the previous rank's displacement plus the number
        // of cells the previous rank will process
        for(i=1;i<num_ranks;i++) {
            displs[i] = displs[i-1] + recv_counts[i-1];
        }
    }

    // Broadcast each element of the structs, to avoid the complexity
    // of creating custom data types
    MPI_Bcast(&m1.rows,1,MPI_INT,FIRST_RANK,MPI_COMM_WORLD);
    MPI_Bcast(&m1.cols,1,MPI_INT,FIRST_RANK,MPI_COMM_WORLD);
    MPI_Bcast(&m2.rows,1,MPI_INT,FIRST_RANK,MPI_COMM_WORLD);
    MPI_Bcast(&m2.cols,1,MPI_INT,FIRST_RANK,MPI_COMM_WORLD);
    MPI_Bcast(m1.matrix,m1.rows*m1.cols,MPI_INT,FIRST_RANK,MPI_COMM_WORLD);
    MPI_Bcast(m2.matrix,m2.rows*m2.cols,MPI_INT,FIRST_RANK,MPI_COMM_WORLD);
    MPI_Bcast(row_counts,num_ranks,MPI_UNSIGNED,FIRST_RANK,MPI_COMM_WORLD);
    MPI_Bcast(recv_counts,num_ranks,MPI_UNSIGNED,FIRST_RANK,MPI_COMM_WORLD);

    

    // Calculate row offset in product matrix to start and stop calculation
    row_stride              = row_counts[rank];
    // The first rank starts at row 0
    start_row               = 0;
    // Each subsequent rank will start at the row equal to the sum of all
    // the previous ranks' row counts
    for(i=0;i<rank;i++) {
        start_row += row_counts[i];
    }
    // Each rank will stop after processing the rows between start_row and
    // row_stride
    stop_row                = start_row+row_stride;

    // Only need to allocate a matrix big enough for the local computation
    local_dst_m.cols        = m2.cols;
    local_dst_m.rows        = row_stride;
    local_cells             = recv_counts[i];
    local_dst_m.matrix      = safe_malloc_int(
            local_cells,
            "Allocating local destination matrix"
            );

#ifdef DEBUG
    fprintf(stderr,"Rank %d - Row stride %u, start_row %u, stop_row %u\n",
            rank,row_stride,start_row,stop_row);
#endif
    matmul(&m1,&m2,&local_dst_m,start_row,stop_row);

#ifdef DEBUG
    fprintf(stderr,"Rank %d local destination matrix:\n",rank);
    print_matrix(&local_dst_m);
    if(rank == FIRST_RANK) {
        fprintf(stderr,"Rank %d full size: %u\n",rank,(dst_m.cols*dst_m.rows));
        fprintf(stderr,"Rank %d gathering to %u cells\n",rank,dst_cells);
        fprintf(stderr,"Displacents:\n");
        for(i=0;i<num_ranks;i++) {
            fprintf(stderr,"%d\n",displs[i]);
        }
    }
    fprintf(stderr,"Rank %d gathering from %u cells\n",rank,local_cells);
#endif

    // Gather the local destination matrices into the first rank's
    // full-size destination matrix
    MPI_Gatherv(
            local_dst_m.matrix,
            local_cells,
            MPI_INT,
            dst_m.matrix,
            recv_counts,
            displs,
            MPI_INT,
            FIRST_RANK,
            MPI_COMM_WORLD
            );

    if(rank == FIRST_RANK && print) {
        fprintf(stderr,"Destination matrix:\n");
        print_matrix(&dst_m);
    }

    MPI_Finalize();

    exit(EXIT_SUCCESS);
}
