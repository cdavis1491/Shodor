/* Parallelization:  Sieve of Eratosthenes
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * January 2012
 * Modified: Mobeen
 * May 2016
 * MPI code
 *  How to compile:
 *      $ make mpi_sieve
 *  How to Run: 
 *      $ aprun -n p ./mpi_sieve -n N
 *     where p is the number of  processes and N is the value 
 *     under which to find primes.
 ******************************************************************/
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    /* Declare variables */
    int N = 50; /* The positive integer under which we are finding primes */
    int sqrtN = 0; /* The square root of N, which is stored in a variable to 
                      avoid making excessive calls to sqrt(N) */
    int c = 0;   /* Used to check the next number to be circled */
    int m = 0;  /* Used to check the next number to be marked */
    
    int *list1; /* The list of numbers <= sqrtN -- if list1[x] equals 1, then x 
                   is marked.  If list1[x] equals 0, then x is unmarked. */
    
    int *list2; /* The list of numbers > sqrtN -- if list2[x-lowest_num] is marked, then 
                   x is marked.  If list2[x-lowest_num] equals 0, then x is unmarked. */
    
    char next_option = ' '; /* Used for parsing command line arguments */
    int S = 0; /* A near-as-possible even split of the count of numbers above 
                  sqrtN */
    int remainder = 0; /* The remainder of the near-as-possible even split */
    int lowest_num = 0; /* The lowest number in the current process's split */
    int highest_num = 0; /* The highest number in the current process's split */
    int my_rank = 0; /* The rank of the current process */
    int num_procs = 0; /* The total number of processes */
	int max = 0; /* The maximum size needed for list2 */
    
    /* Initialize the MPI Environment */
    MPI_Init(&argc, &argv);
    /* Determine the rank of the current process and the number of processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   
    /* Parse command line arguments -- enter 'man 3 getopt' on a shell to see
       how this works */
    while((next_option = getopt(argc, argv, "n:")) != -1) {
        switch(next_option) {
            case 'n':
                N = atoi(optarg);
                break;
            case '?':
            default:
                fprintf(stderr, "Usage: %s [-n N]\n", argv[0]);
                exit(-1);
        }
    }

    /* Calculate sqrtN */
    sqrtN = (int)sqrt(N);

    /* Calculate S, remainder, lowest_num, and highest_num */
    S = (N-(sqrtN+1)) / num_procs;
    remainder = (N-(sqrtN+1)) % num_procs;
    lowest_num = sqrtN + my_rank*S + 1;
    
    highest_num = lowest_num+S-1;
    
    if(my_rank == num_procs-1) {
        highest_num += remainder;
    }
	
	max = highest_num - lowest_num + 1 + remainder;
    
    /* Allocate memory for lists */
    list1 = (int*)malloc((sqrtN+1) * sizeof(int));
    list2 = (int*)malloc(max * sizeof(int));

    /* Exit if malloc failed */
    if(list1 == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for list1.\n");
        exit(-1);
    }
    if(list2 == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for list2.\n");
        exit(-1);
    }

    /* Run through each number in list1 */
    for(c = 2; c <= sqrtN; c++) {

        /* Set each number as unmarked */
        list1[c] = 0;
    }
    
    /* Run through each number in list2 */
    for(c = lowest_num; c <= highest_num; c++) {

        /* Set each number as unmarked */
        list2[c-lowest_num] = 0;
    }

    /* Run through each number in list1 */
    for(c = 2; c <= sqrtN; c++) {

        /* If the number is unmarked */
        if(list1[c] == 0) {

            /* Run through each number bigger than c in list1 */
            for(m = c+1; m <= sqrtN; m++) {

                /* If m is a multiple of c */
                if(m%c == 0) {

                    /* Mark m */
                    list1[m] = 1;
                }
            }

            /* Run through each number bigger than c in list2 */
            for(m = lowest_num; m <= highest_num; m++)
            {
                /* If m is a multiple of C */
                if(m%c == 0)
                {
                    /* Mark m */
                    list2[m-lowest_num] = 1;
                }
            }
        }
    }
    int newline = 0;
    printf("prime numbers in list1 are: \n");
    /* If Rank 0 is the current process */
    if(my_rank == 0) {

        /* Run through each of the numbers in list1 */
        for(c = 2; c <= sqrtN; c++) {

            /* If the number is unmarked */
            if(list1[c] == 0) {
                /* The number is prime, print it */
                printf("\t%d", c);
                if(newline % 10 == 0){
                    printf("\n");
                }
            }
        }
    
        printf("prime numbers in list2 are: \n");
        /* Run through each of the numbers in list2 */
        for(c = lowest_num; c <= highest_num; c++) {

            /* If the number is unmarked */
            if(list2[c-lowest_num] == 0) {

                /* The number is prime, print it */
                printf("%d ", c);
                if(newline % 10 == 0){
                    printf("\n");
                }
            }
        }

        /* Run through each of the other processes */
        for(my_rank = 1; my_rank <= num_procs-1; my_rank++) {
            
            /* Calculate lowest_num and highest_num for r */
            lowest_num = sqrtN + my_rank*S + 1;
            highest_num = lowest_num+S-1;
            if(my_rank == num_procs-1) {
                highest_num += remainder;
            }
            
            /* Receive list2 from the process */
            MPI_Recv(list2, highest_num-lowest_num+1, MPI_INT, my_rank, 0, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);

            /* Run through the list2 that was just received */
            for(c = lowest_num; c <= highest_num; c++) {

                /* If the number is unmarked */
                if(list2[c-lowest_num] == 0) {

                    /* The number is prime, print it */
                    printf("%d ", c);
                }
            }
        }
        printf("\n");

        /* If the process is not Rank 0 */
    } else {

        /* Send list2 to Rank 0 */
        MPI_Send(list2, highest_num-lowest_num+1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    /* Deallocate memory for list */
    free(list2);
    free(list1);

    /* Finalize the MPI environment */
    MPI_Finalize();

    return 0;
}
