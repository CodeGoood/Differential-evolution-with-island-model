#ifndef _F4_H
#define _F4_H

int nextInt(int n);
int next(int bits);
double nextDouble();
double nextGaussian();
int* createPermVector(int dim, double min,double max);
double* readR(int sub_dim, int BENCH);
__host__ __device__ void multiply(double *vector, double *matrix, int dim, double *anotherz1);
__host__ __device__ void rotateVector(int i, int *c, double *anotherz1, int *s, double *anotherz, int *Pvector, double *r25, double *r50, double *r100);
__host__ __device__ double F4(double *x, double *Ovector, double *anotherz, int *Pvector, double *r25, double *r50, double *r100);


#endif

