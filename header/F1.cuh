#ifndef _F1_H
#define _F1_H

__host__ __device__ int sign(double x);
__host__ __device__ double hat(double x);
__host__ __device__ double c1(double x);
__host__ __device__ double c2(double x);
__host__ __device__ void transform_osz(double* z, int dim);
__host__ __device__ double elliptic(double*x,int dim);
__host__ __device__ double F1(double *x, double *Ovector, double *anotherz);

#endif
