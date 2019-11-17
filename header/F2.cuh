#ifndef _F2_H
#define _F2_H

__host__ __device__ void Lambda(double* z, double alpha, int dim);
__host__ __device__ void transform_asy(double* z, double beta, int dim);
__host__ __device__ double rastrigin(double*x,int dim);
__host__ __device__ double F2(double *x, double *Ovector, double *anotherz);

#endif
