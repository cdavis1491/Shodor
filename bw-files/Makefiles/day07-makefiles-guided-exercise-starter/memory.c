#include "memory.h"

/* DECLARE FUNCTIONS */

/* Allocate dynamic memory */
void AllocateMemory() {
  Trees    = (int*)malloc(NTreesPlusBounds * sizeof(int));
  NewTrees = (int*)malloc(          NTrees * sizeof(int));
}

/* Free allocated memory */
void FreeMemory() {
  free(NewTrees);
  free(Trees);
}

