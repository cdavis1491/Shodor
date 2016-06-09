#include "output.h"

/* Output tree data for the current time step */
void OutputData() {
  int row;
  int col;
  char buf[64];

  /* Write the header for the current time step */
  sprintf(buf, "Time step %d\n", CurStep);
  fprintf(OutputFile, "%s", buf);

  for (row = 1; row < NRows + 1; row++) {
    for (col = 1; col < NCols + 1; col++) {
      sprintf(buf, "%*d ",
          NMaxBurnStepsDigits, Trees[TREE_MAP(row, col, NColsPlusBounds)]);
      fprintf(OutputFile, "%s", buf);
    }
    fprintf(OutputFile, "\n");
  }

  /* Write the newline between time steps */
  fprintf(OutputFile, "\n");
}

