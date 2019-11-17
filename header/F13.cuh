#ifndef _F13_H
#define _F13_H

__host__ __device__ double rotateVectorConform(int i, int *c, double *anotherz1, int *s, double *anotherz, int *Pvector, double *r25, double *r50, double *r100);
__host__ __device__ double F13(double *x, double *Ovector, double *anotherz, int *Pvector, double *r25, double *r50, double *r100);


#endif

