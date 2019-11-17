#ifndef _DIV_H
#define _DIV_H

__global__ void measure_div(double *pop, double *var, int Np, int Ni, int N, int island_id);
__global__ void measure_div_total(double *pop, double *var, int Np, const int Ni, int N);

#endif
