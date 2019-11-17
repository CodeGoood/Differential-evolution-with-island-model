#include <stdio.h>

__global__ void measure_div(double *pop, double *var, int Np, const int Ni, int N, int island_id)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x + y * blockDim.x * gridDim.x;

    double avg = 0.f;
    var[id] = 0.f;

    for(int i = 0; i < Np; i++)
        avg += pop[id + island_id*Np + i*N];
    avg /= Np;
    
    for(int i = 0; i < Np; i++)
        var[id] += (pop[id + island_id*Np + i*N] - avg) * (pop[id + island_id*Np + i*N] - avg);
    var[id] /= Np;
    var[id] = sqrt(var[id]);
}

__global__ void measure_div_total(double *pop, double *var, int Np, const int Ni, int N)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x + y * blockDim.x * gridDim.x;

    double avg = 0.f;
    var[id] = 0.f;

    for(int i = 0; i < Np*Ni; i++)
        avg += pop[id + i*N];
    avg /= Np*Ni;

    for(int i = 0; i < Np*Ni; i++)
        var[id] += (pop[id + i*N] - avg) * (pop[id + i*N] - avg);
    var[id] /= Np*Ni;
    var[id] = sqrt(var[id]);
}
