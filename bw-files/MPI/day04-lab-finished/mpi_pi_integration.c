/*******************************************************************************
 * This is a simple program that shows the numerical integration example.
 * It computes pi by approximating the area under the curve:
 *      f(x) = 4 / (1+x*x) between 0 and 1.
 * To do this intergration numerically, the interval from 0 to 1 is divided into
 * some number (num_sub_intervals) subintervals and added up the area of
 * rectangles
 * The larger the value of the num_sub_interval the more accurate your result
 * will be.
 *
 * The program first asks the user to input a value for subintervals, it
 * computes the approximation for pi, and then compares it to a more accurate aproximate 
 * value of pi in the math.h library.
 *
 * This program is parallelized using MPI collective communication operations such as:
 * 		- MPI_Bcast(...): when one process sends message to all other process in the pool
 *		- MPI_Reduce(...): when one process recvies/collects messages from every one else 
 *
 * Author: Mobeen Ludin
 *
 * How to Setup Runtime Environment:
 *  $ qsub -I -l nodes=2:ppn=32:xe,walltime=01:30:00 -l advres=bwintern
 *
 * How to Compile:
 *	$ cc mpi_pi_integration.c -o mpi_pi_integration 
 *
 * How to Run:
 *	$ aprun -n 8 ./mpi_pi_integration
 *
 * OutPut: make sure you run the program on multiple codes (-n ##). Also 
 * evrytime it ask you for the number of itteration. start with 1000, and add
 * a zero every other time you run the program.
 *
 ******************************************************************************/
// The following C libraries are needed.
#include <stdio.h>    // Is needed for printing the final results
#include <stdlib.h>   // Is needed for exiting early if an error occurs
#include <math.h>     // Is needed for fabs()/absolute value of floating point numbers
#include <mpi.h> 	 // Is needed for MPI parallelization

int main(int argc, char *argv[]) {
	int num_sub_intervals = 0;  	// number of sub intervals
	double start_time, end_time, time_diff; // for timeing most computation intese part of the program
	double x, pi, pi_tot;	   // pi=value of pi, pi_total= total value of pi after reduction
	double sum = 0.0; // for partial sum
	double step; 
	int i;
	
	// MPI Initialization part
	int pRank;	//every process will have a unique integer id/rank in the pool
	int pNum; 	// How many processes are in the pool
	// MPI_Init(...) starts the parallel regions, here on the part of the code until
	// MPI_Finalize(...) call will be executed by all processes in the pool
	MPI_Init(&argc, &argv);
	//Next line will setup the number of processes in the communicator/pool. besed on -n ##
	MPI_Comm_size(MPI_COMM_WORLD, &pNum);
	//Next line will give a unique id to each process in the communicator/pool of processes
	MPI_Comm_rank(MPI_COMM_WORLD, &pRank);

	if(pRank == 0){  // If I am process 0 then ask the user for an input (number of intervals)
		printf("Please enter the number of iterations used to compute pi: \n");
		scanf("%d", &num_sub_intervals);
	}
	//Sending the number of sub intervals to all process in the communicator
	MPI_Bcast(&num_sub_intervals, 1, MPI_INT, 0, MPI_COMM_WORLD);

	step = 1.0/(double) num_sub_intervals;  

	// Record the start time from here on:
	start_time = MPI_Wtime();

	for(i=pRank+1; i<=num_sub_intervals; i+=pNum){
		x = step*((double)i-0.5);
		sum += 4.0/(1.0+x*x);
	}
	pi = step * sum;

	//after each process is done with calculating their partial sum. the process 0 will ask all
	// other proceeses in the pool for their partial sums, and add them together to make a total
	// sum/value of pi.
	MPI_Reduce(&pi, &pi_tot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	//End recording the time here.
	end_time = MPI_Wtime();
	time_diff = end_time - start_time;

	if(pRank == 0){ //If I am process 0, prints out the results to user.
		// print the result here:
		printf("computed pi value is = %g (%17.15f)\n\n", pi_tot,pi_tot);
		printf("M_PI accurate value from math.h is: %17.15f \n\n", M_PI);
		printf("Difference between computed pi and math.h M_PI = %17.15f\n\n",
				fabs(pi_tot - M_PI));
		printf("Time to compute = %g seconds\n\n", time_diff);
	}
	//Line is for end of parallel regions of the code.
	MPI_Finalize();
	return EXIT_SUCCESS;
}

