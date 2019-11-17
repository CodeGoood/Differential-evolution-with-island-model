#ifndef _F14_H
#define _F14_H

__host__ __device__ double rotateVectorConflict(int i, int *c, double *x, double *anotherz1, int *s, double *anotherz, int *Pvector, double *r25, double *r50, double *r100);
__host__ __device__ double F14(double *x, double *Ovector, double *anotherz, int *Pvector, double *r25, double *r50, double *r100);

#endif

