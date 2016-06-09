/**************************************************************
 * Filename: omp_critical.c
 * Description: first intro to OpenMP directives
 * How to compile:
 *  $ cc omp_critical.c -o omp_critical
 * How to set number of threads to 8:
 *  $ export OMP_NUM_THREADS=8
 * How to Run: 
 *  $ aprun -n 1 -d 8 ./omp_critical 
 *************************************************************/
#include <stdio.h>	 // Is needed for printing the final results
#include <stdlib.h>  // Is needed for exiting early if an error occurs
#include <math.h>   // Is needed for fabs()/absolute value of floating point numbers
#include <omp.h>

int main(int argc, char *argv[]) {
    int num_rect = 0;   // number of rectangles
    double x_midp, pi;
    double sum = 0.0;
    double rect_width;
	int i;
    double start_t, end_t, compute_t;

    printf("Please enter the number of rectangles to compute pi: \n");
    scanf("%d",&num_rect);

    rect_width = 1.0/(double)num_rect;
    
    start_t = omp_get_wtime();
#pragma omp parallel default(none) shared(x_midp,rect_width,num_rect,sum) private(i)
    {
    #pragma omp for
        for(i=0; i < num_rect; i++){
            #pragma omp critical
            {
                x_midp = (i+0.5)*rect_width;
                sum += 4.0/(1.0+x_midp*x_midp);
            }
        }
    } // END: pragma
    pi = rect_width * sum;
    
    end_t = omp_get_wtime();
    compute_t = end_t - start_t;

    // print the result here: 
    printf("Results are: \n");
    printf("\t Computed pi is: %g (%17.15f)\n\n", pi,pi);
    printf("\t M_PI value from math.h is: %17.15f \n\n", M_PI);
    printf("\t Difference between pi and M_PI = %17.15f\n\n",
            fabs(pi - M_PI));
    printf("\t Total time to compute the for loop was: %lf \n", compute_t);
    return EXIT_SUCCESS;
}

