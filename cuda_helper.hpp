/* Benoit LAGADEC 

For more information refer to this code 

https://github.com/NVIDIA/cuda-samples

*/


#pragma once

#include <cstdio>

// Define some error checking macros.
#define cudaErrCheck(stat)                                                                                             \
  {                                                                                                                    \
    cudaErrCheck_((stat), __FILE__, __LINE__);                                                                         \
  }
inline void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
  if (stat != cudaSuccess) { fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line); }
}

