#ifndef _F12_H
#define _F12_H

__host__ __device__ double rosenbrock(double*x,int dim);
__host__ __device__ double F12(double *x, double *Ovector, double *anotherz, int *Pvector, double *r25, double *r50, double *r100);

#endif

