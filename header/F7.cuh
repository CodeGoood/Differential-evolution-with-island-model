#ifndef _F7_H
#define _F7_H

__host__ __device__ double schwefel(double*x,int dim);
__host__ __device__ double F7(double *x, double *Ovector, double *anotherz, int *Pvector, double *r25, double *r50, double *r100);

#endif

