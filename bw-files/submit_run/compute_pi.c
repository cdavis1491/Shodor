/*******************************************************************************
 * This is a simple program that shows the numerical integration example.
 * It computes pi by approximating the area under the curve:
 *      f(x) = 4 / (1+x*x) between 0 and 1.
 * To do this intergration numerically, the interval from 0 to 1 is divided into
 * some number (num_rect) subintervals and added up the area of
 * rectangles
 * The larger the value of the num_rect the more accurate your result
 * will be.
 *
 * The program first asks the user to input a value for subintervals, it
 * computes the approximation for pi, and then compares it to a more 
 * accurate aproximate value of pi in the math.h library.
 *
 * Hwo to Compile:
 *  $ gcc compute_pi.c -o compute_pi.exe
 * How to Run:
 *  $ ./compute_pi.exe
 ******************************************************************************/
// The following C libraries are needed.
#include <stdio.h>	 // Is needed for printing the final results
#include <stdlib.h>  // Is needed for exiting early if an error occurs
#include <math.h>   // Is needed for fabs()/absolute value of floating point numbers

int main(int argc, char *argv[]) {
    int num_rect = 0;   // number of rectangles
    double x_midp, pi;
    double sum = 0.0;
    double rect_width;
	int i;

    printf("Please enter the number of rectangles to compute pi: \n");
    scanf("%d",&num_rect);

    rect_width = 1.0/(double)num_rect;  //width of individual rectangle

    for(i=0; i < num_rect; i++){
        x_midp = (i+0.5)*rect_width;    //Compute the hiegh of each rectangle.
        sum += 4.0/(1.0+x_midp*x_midp); //compute area of each rectangle
    }
    pi = rect_width * sum;

    // print the result here: 
    printf("computed pi value is = %g (%17.15f)\n\n", pi,pi);
    printf("M_PI accurate value from math.h is: %17.15f \n\n", M_PI);
    printf("Difference between computed pi and math.h M_PI = %17.15f\n\n",
            fabs(pi - M_PI));
    return EXIT_SUCCESS;
}

