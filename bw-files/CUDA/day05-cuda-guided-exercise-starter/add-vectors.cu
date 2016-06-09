// This is a CUDA program that does the following:
//
// 1. On the host, fill the A and B arrays with random numbers
// 2. On the host, print the initial values of the A and B arrays
// 3. Copy the A and B arrays from the host to the device
// 4. On the device, add the A and B vectors and store the result in C
// 5. Copy the C array from the device to the host
// 6. On the host, print the result
//
// Author: Aaron Weeden, Shodor, 2016

// Import library so we can call printf()
#include <stdio.h>

// Import library so we can call exit(), malloc(), free(), random(), etc.
#include <stdlib.h>

// Import library so we can call time()
#include <time.h>

// Define the number of numbers in each array
#define NUM_COUNT 10

// Define the number of bytes in each array
#define BYTE_COUNT ((NUM_COUNT) * sizeof(int))

// Declare functions that will be defined later
void TryMalloc(void * const err);

// Start the program
int main()
{
  // Declare variables for the host and device arrays
  int * hostA;
  int * hostB;
  int * hostC;
  int * deviceA;
  int * deviceB;
  int * deviceC;

  // Allocate memory for the host arrays

  // Allocate memory for the device arrays

  // Initialize the random number generator
  srandom(time(NULL));

  // On the host, fill the A and B arrays with random numbers
  printf("Expected Result:\n");
  for (int i = 0; i < NUM_COUNT; i++)
  {
    hostA[i] = 100 * random() / RAND_MAX;
    hostB[i] = 100 * random() / RAND_MAX;
    printf("\thostC[%d] should be %d + %d\n", i, hostA[i], hostB[i]);
  }

  // Copy the A and B arrays from the host to the device

  // On the device, add the A and B vectors and store the result in C

  // Copy the C array from the device to the host

  // On the host, print the result
  printf("Result:\n");
  for (int i = 0; i < NUM_COUNT; i++)
  {
    printf("\thostC[%d] = %d\n", i, hostC[i]);
  }

  // De-allocate memory for the device arrays

  // De-allocate memory for the host arrays

  return 0;
}

// Define a function to check whether a malloc() call was successful
void TryMalloc(void * const err)
{
  if (err == NULL)
  {
    fprintf(stderr, "malloc error\n");
    exit(EXIT_FAILURE);
  }
}

