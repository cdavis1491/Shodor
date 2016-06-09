// Checks if the return to malloc() is NULL, and exits if it is
void TryMalloc(void * const err)
{
  if (err == NULL)
  {
    fprintf(stderr, "malloc error\n");
    exit(EXIT_FAILURE);
  }
}

// Checks if the return to sprintf() is negative, and exits if it is
void TrySprintf(int const err)
{
  if (err < 0)
  {
    fprintf(stderr, "ERROR in sprintf\n");
    exit(EXIT_FAILURE);
  }
}

// Checks if the return to memcpy() is NULL, and exits if it is
void TryMemcpy(void * const err)
{
  if (err == NULL)
  {
    fprintf(stderr, "ERROR in memcpy\n");
    exit(EXIT_FAILURE);
  }
}

