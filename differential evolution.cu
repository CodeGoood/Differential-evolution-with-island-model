#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <fstream>
#include <sstream>
#include <thrust/device_ptr.h>
#include <limits.h>
#include <math.h>
#include <cub/cub.cuh>
#include "diversity.cuh"
#include "F1.cuh"
#include "F2.cuh"
#include "F3.cuh"
#include "F4.cuh"
#include "F5.cuh"
#include "F6.cuh"
#include "F7.cuh"
#include "F8.cuh"
#include "F9.cuh"
#include "F10.cuh"
#include "F11.cuh"
#include "F12.cuh"
#include "F13.cuh"
#include "F14.cuh"
#include "F15.cuh"


__host__ __device__ long long int iDivUp(long long int a, long long int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
__host__ __device__ int mod(int a,int b){ return ((a % b + b) %b); }

#define N ((int)1000)
int  DIM;
int IL_DIM;

int SUB_MODEL, MODEL, Np, Ni, ANp;
int MF, DEVICE_ID;
double MR;
#define FE ((long long int)30000000)


#define H ((int)6)
#define p_rate ((double)0.11)
#define arc_rate ((double)2.6)
#define RUNTIME ((int)10)

#define PI 3.14159265358979323846
#define E  2.718281828459045235360287471352
#define L(i) ((int64_t)i)
#define D(i) ((double)i)
double LB;
double UB;

#define the CUDA_API_PER_THREAD_DEFAULT_STREAM
#define gpuErrchk(ans) { HANDLE_ERROR((ans), __FILE__, __LINE__); }
inline void HANDLE_ERROR(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        printf( "ERROR: %s in %s at line %d\n", cudaGetErrorString( code ),file, line );
        exit(0);
    }
}

using namespace thrust;
using namespace std;

inline double randDouble() {
    return (double)rand() / (double) RAND_MAX;
}

inline double cauchy_g(double mu, double gamma) {
    return mu + gamma * tan(M_PI*(randDouble() - 0.5));
}

inline double gauss(double mu, double sigma){
    return mu + sigma * sqrt(-2.0 * log(randDouble())) * sin(2.0 * M_PI * randDouble());
}

__global__ void curand_setup_kernel(curandState * __restrict state, const unsigned long int seed, int Np, int Ni)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < Ni*Np*N) curand_init(seed, id, 0, &state[id]);
}

__global__ void init_pop(double *pop, curandState * __restrict state, int Np, int Ni, double LB, double UB)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < Ni*Np*N)
        pop[id] = (UB - LB) * curand_uniform_double(&state[id]) + LB;
}

__global__ void init_parameter(int *pop_sequence, int Np, int Ni)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < Ni*Np)
        pop_sequence[id] = id;

}

__global__ void init_offsets(int *offsets, int Np, int Ni)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x + y * blockDim.x * gridDim.x;

    if(id <= Ni)
        offsets[id] = id * Np;
}

__device__ int abc_id_fix_compact(const int num, int j, int i, int island_id, int offset, int DIM, int Np, int Ni)
{
    if(num==0) return mod(j-1,DIM) + mod(i-1,DIM) * DIM + island_id * Np;
    if(num==1) return mod(j,DIM)   + mod(i-1,DIM) * DIM + island_id * Np;//j + mod(i-1,DIM) * DIM + island_id * Np;
    if(num==2) return mod(j+1,DIM) + mod(i-1,DIM) * DIM + island_id * Np;//mod(j-1,DIM) * offset + mod(i+1,DIM) + island_id * DIM;
    if(num==3) return mod(j-1,DIM) + mod(i,DIM)   * DIM + island_id * Np;//j * offset + mod(i-1,DIM) + island_id * DIM;
    if(num==4) return mod(j+1,DIM) + mod(i,DIM)   * DIM + island_id * Np;//j * offset + mod(i+1,DIM) + island_id * DIM;
    if(num==5) return mod(j-1,DIM) + mod(i+1,DIM) * DIM + island_id * Np;//mod(j+1,DIM) * offset + mod(i-1,DIM) + island_id * DIM;
    if(num==6) return mod(j,DIM)   + mod(i+1,DIM) * DIM + island_id * Np;//mod(j+1,DIM) * offset + i;
    if(num==7) return mod(j+1,DIM) + mod(i+1,DIM) * DIM + island_id * Np;//mod(j+1,DIM) * offset + mod(i+1,DIM) + island_id * DIM;
}

__device__ int abc_id_fix_compact_island(const int num, int j, int i, int IL_DIM, int Np, int Ni)
{
    if(num==0) return mod(j-1,IL_DIM) + mod(i-1,IL_DIM) * IL_DIM;
    if(num==1) return mod(j,IL_DIM)   + mod(i-1,IL_DIM) * IL_DIM;//j + mod(i-1,DIM) * DIM + island_id * Np;
    if(num==2) return mod(j+1,IL_DIM) + mod(i-1,IL_DIM) * IL_DIM;//mod(j-1,DIM) * offset + mod(i+1,DIM) + island_id * DIM;
    if(num==3) return mod(j-1,IL_DIM) + mod(i,IL_DIM)   * IL_DIM;//j * offset + mod(i-1,DIM) + island_id * DIM;
    if(num==4) return mod(j+1,IL_DIM) + mod(i,IL_DIM)   * IL_DIM;//j * offset + mod(i+1,DIM) + island_id * DIM;
    if(num==5) return mod(j-1,IL_DIM) + mod(i+1,IL_DIM) * IL_DIM;//mod(j+1,DIM) * offset + mod(i-1,DIM) + island_id * DIM;
    if(num==6) return mod(j,IL_DIM)   + mod(i+1,IL_DIM) * IL_DIM;//mod(j+1,DIM) * offset + i;
    if(num==7) return mod(j+1,IL_DIM) + mod(i+1,IL_DIM) * IL_DIM;//mod(j+1,DIM) * offset + mod(i+1,DIM) + island_id * DIM;
}

__device__ int abc_ring(const int num, int island, int Np, int Ni)
{
    if(num==0)
    {
        island-=1;
        if(island < 0)
            return Ni-1;
        else
            return island;
    }
    if(num==1) 
    {
        island+=1;
        if(island==Ni)
            return 0;
        else
            return island;
    }
}

__device__ int abc_id_fix_compact_dif_island(const int num, int j, int i, int IL_DIM, int Np, int Ni, int SUB_MODEL)
{
    if(SUB_MODEL==0)
    {
        if(num==0) return mod(j-1,11) + mod(i-1,12) * 11;
        if(num==1) return mod(j,11)   + mod(i-1,12) * 11;//j + mod(i-1,DIM) * DIM + island_id * Np;
        if(num==2) return mod(j+1,11) + mod(i-1,12) * 11;//mod(j-1,DIM) * offset + mod(i+1,DIM) + island_id * DIM;
        if(num==3) return mod(j-1,11) + mod(i,12)   * 11;//j * offset + mod(i-1,DIM) + island_id * DIM;
        if(num==4) return mod(j+1,11) + mod(i,12)   * 11;//j * offset + mod(i+1,DIM) + island_id * DIM;
        if(num==5) return mod(j-1,11) + mod(i+1,12) * 11;//mod(j+1,DIM) * offset + mod(i-1,DIM) + island_id * DIM;
        if(num==6) return mod(j,11)   + mod(i+1,12) * 11;//mod(j+1,DIM) * offset + i;
        if(num==7) return mod(j+1,11) + mod(i+1,12) * 11;//mod(j+1,DIM) * offset + mod(i+1,DIM) + island_id * DIM;
    }

    if(SUB_MODEL==2)
    {
        if(num==0) return mod(j-1,22) + mod(i-1,23) * 22;
        if(num==1) return mod(j,22)   + mod(i-1,23) * 22;//j + mod(i-1,DIM) * DIM + island_id * Np;
        if(num==2) return mod(j+1,22) + mod(i-1,23) * 22;//mod(j-1,DIM) * offset + mod(i+1,DIM) + island_id * DIM;
        if(num==3) return mod(j-1,22) + mod(i,23)   * 22;//j * offset + mod(i-1,DIM) + island_id * DIM;
        if(num==4) return mod(j+1,22) + mod(i,23)   * 22;//j * offset + mod(i+1,DIM) + island_id * DIM;
        if(num==5) return mod(j-1,22) + mod(i+1,23) * 22;//mod(j+1,DIM) * offset + mod(i-1,DIM) + island_id * DIM;
        if(num==6) return mod(j,22)   + mod(i+1,23) * 22;//mod(j+1,DIM) * offset + i;
        if(num==7) return mod(j+1,22) + mod(i+1,23) * 22;//mod(j+1,DIM) * offset + mod(i+1,DIM) + island_id * DIM;
    }

}

__host__ __device__ double fitness(double *pop)
{
    double sum = 0.f;
    for(int i = 0; i < 1000; i++)
        sum += pop[i] * (sin(sqrt(fabs(pop[i]))));
    return (418.98291*1000 - sum);
}

__global__ void fitness_GPU(double *pop, double *fit, int Np, int Ni, double *Ovector, double *d_anotherz, int Bench, int *Pvector, double *OvectorVec, double *r25,double *r50, double *r100)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x + y * blockDim.x * gridDim.x;

    if(id < Np*Ni && Bench==1)  fit[id] =  F1(&pop[id * N], Ovector, &d_anotherz[id * N]);
    if(id < Np*Ni && Bench==2)  fit[id] =  F2(&pop[id * N], Ovector, &d_anotherz[id * N]);
    if(id < Np*Ni && Bench==3)  fit[id] =  F3(&pop[id * N], Ovector, &d_anotherz[id * N]);
    if(id < Np*Ni && Bench==4)  fit[id] =  F4(&pop[id * N], Ovector, &d_anotherz[id * N], Pvector,r25,r50,r100);
    if(id < Np*Ni && Bench==5)  fit[id] =  F5(&pop[id * N], Ovector, &d_anotherz[id * N], Pvector,r25,r50,r100);
    if(id < Np*Ni && Bench==6)  fit[id] =  F6(&pop[id * N], Ovector, &d_anotherz[id * N], Pvector,r25,r50,r100);
    if(id < Np*Ni && Bench==7)  fit[id] =  F7(&pop[id * N], Ovector, &d_anotherz[id * N], Pvector,r25,r50,r100);
    if(id < Np*Ni && Bench==8)  fit[id] =  F8(&pop[id * N], Ovector, &d_anotherz[id * N], Pvector,r25,r50,r100);
    if(id < Np*Ni && Bench==9)  fit[id] =  F9(&pop[id * N], Ovector, &d_anotherz[id * N], Pvector,r25,r50,r100);
    if(id < Np*Ni && Bench==10) fit[id] = F10(&pop[id * N], Ovector, &d_anotherz[id * N], Pvector,r25,r50,r100);
    if(id < Np*Ni && Bench==11) fit[id] = F11(&pop[id * N], Ovector, &d_anotherz[id * N], Pvector,r25,r50,r100);
    if(id < Np*Ni && Bench==12) fit[id] = F12(&pop[id * N], Ovector, &d_anotherz[id * N], Pvector,r25,r50,r100);
    if(id < Np*Ni && Bench==13) fit[id] = F13(&pop[id * N], Ovector, &d_anotherz[id * N], Pvector,r25,r50,r100);
    if(id < Np*Ni && Bench==14) fit[id] = F14(&pop[id * N], OvectorVec, &d_anotherz[id * N], Pvector,r25,r50,r100);
    if(id < Np*Ni && Bench==15) fit[id] = F15(&pop[id * N], Ovector, &d_anotherz[id * N], Pvector,r25,r50,r100);

}

__global__ void generate_rand_cellular(double *RRand, curandState * __restrict state, int Np, int Ni, int DIM)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x + y * blockDim.x * gridDim.x;

    if(id<Np*Ni)
    {
        int a,b,c;

        a = curand(&state[id])%8;
        while(a==id) a =  curand(&state[id])%8;
        RRand[id*3] = abc_id_fix_compact(a, id%DIM, id/DIM, id/(DIM*DIM), Ni*DIM, DIM, Np, Ni);

        b =  curand(&state[id])%8;
        while(b==a || b==id) b =  curand(&state[id])%8;
        RRand[id*3] = abc_id_fix_compact(a, id%DIM, id/DIM, id/(DIM*DIM), Ni*DIM, DIM, Np, Ni);

        c =  curand(&state[id])%8;
        while(c==a || c==b || c==id)  c = curand(&state[id])%8;
        RRand[id*3] = abc_id_fix_compact(a, id%DIM, id/DIM, id/(DIM*DIM), Ni*DIM, DIM, Np, Ni);
    }
}

__global__ void generate_rand(double *RRand, curandState * __restrict state, int Np, int Ni)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x + y * blockDim.x * gridDim.x;

    if(id<Np*Ni)
    {
        int a,b,c;

        a = curand(&state[id])%(Np*Ni);
        while(a==id) a =  curand(&state[id])%(Np*Ni);
        RRand[id*3] = a;

        b =  curand(&state[id])%(Np*Ni);
        while(b==a || b==id) b =  curand(&state[id])%(Np*Ni);
        RRand[id*3+1] = b;

        c =  curand(&state[id])%(Np*Ni);
        while(c==a || c==b || c==id)  c = curand(&state[id])%(Np*Ni);
        RRand[id*3+2] = c;
    }
}

__global__ void key_generate_rand(double *keys, curandState * __restrict state, int Np, int Ni)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id<Ni*Np)
        keys[id] = curand_uniform_double(&state[id]);
}

__global__ void recombination(double *pop, double *mutation, double *RRand, curandState * __restrict state, int Np, int Ni, int UB, int LB)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x + y * blockDim.x * gridDim.x;

    if(id < Ni*Np*N)
    {
        int a, b, c;

        a = RRand[(id/N)*3];
        b = RRand[(id/N)*3+1];
        c = RRand[(id/N)*3+2];

        double F = (1.0 - 0.5) * curand_uniform_double(&state[id]) + 0.5;

        if(curand_uniform_double(&state[id]) < 0.9)
        {
            //double F = (1.0 - 0.5) * curand_uniform_double(&state[id]) + 0.5;
            mutation[id] = pop[a*N+(id%N)] + F * (pop[b*N+(id%N)] - pop[c*N+(id%N)]);

            if(mutation[id] < LB || mutation[id] > UB)
                mutation[id] = (UB - LB) * curand_uniform_double(&state[id]) + LB;
        }
    }
}

__global__ void selection(double *pop, double *p_fit, double *mutation, double *m_fit, int Np, int Ni)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x + y * blockDim.x * gridDim.x;

    if(id < Ni*Np)
        if(p_fit[id] > m_fit[id])
        {
            for(int i=0; i<N; i++)
                    pop[id*N+i] = mutation[id*N+i];
            p_fit[id] = m_fit[id];
        }
}

__global__ void migration_ring_r2r(double *pop, double *tmp_pop, double *p_fit, int *shuffle, int *shuffle1, int *pop_size, curandState * __restrict state, double MR, int Np, int Ni)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x + y * blockDim.x * gridDim.x;

    if(id < Ni * pop_size[0])
    {
        int mr = pop_size[0] / MR;
        if(mr==0)
            mr = 1;
        if( id % pop_size[0] < mr)
        {
            for(int k=0; k < N ; k++)
                pop[shuffle[id] * N + k] = tmp_pop[shuffle1[(id+Np) % (Ni*Np)] * N + k];
            p_fit[shuffle[id]] = p_fit[shuffle1[(id+Np) % (Ni*Np)]];
        }
    }
}

__global__ void migration_fully_b2r(double *pop, double *tmp_pop, double *p_fit, int *sorted,  curandState * __restrict state, double MR, int Np, int Ni)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x + y * blockDim.x * gridDim.x;

    if(id < Ni)
    {
        int mr =  MR;
        int send_island;

        for(int i=0; i < mr; i++)
        {
            int a = curand(&state[id])%Np;

            send_island = curand(&state[id])%Ni;
            while(send_island==id)
                send_island = curand(&state[id])%Ni;

            for(int k=0; k<N; k++)
                pop[ ( a + send_island*Np ) * N + k ] = tmp_pop[ sorted[ id*Np ] * N + k ];
            p_fit[  a + send_island*Np  ] = p_fit[ sorted[ id*Np ] ];

        }
    }
}

__global__ void migration_ring_b2r(double *pop, double *tmp_pop, double *p_fit, int *sorted,  curandState * __restrict state, double MR, int Np, int Ni)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x + y * blockDim.x * gridDim.x;

    if(id < Ni)
    {
        int mr =  MR;
        int send_island;
        int over;
        int sec = 5;
        
        for(int i=0; i < mr; i++)
        {
            int a = curand(&state[id])%Np;
            over = curand(&state[id])%2;
            while(over==sec)
                over = curand(&state[id])%2;
            sec = over;

            send_island = abc_ring(over, id, Np, Ni);
            
            for(int k=0; k<N; k++)
                pop[ ( a + send_island*Np ) * N + k ] = tmp_pop[ sorted[ id*Np ] * N + k ];
            p_fit[  a + send_island*Np  ] = p_fit[ sorted[ id*Np ] ];

        }
    }
}

__global__ void migration_island_b2r(double *pop, double *tmp_pop,  double *p_fit, int *sorted, curandState * __restrict state, int Np, int Ni, int DIM, int IL_DIM, int MR)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x + y * blockDim.x * gridDim.x;

    if(id < Ni)
    {
        int come_island;
        int mr = MR;
        
        for(int i=0; i < mr; i++)
        {
            int a = curand(&state[id])%Np;

            come_island = abc_id_fix_compact_island ((curand(&state[id])%8), id%IL_DIM, id/IL_DIM, IL_DIM, Np, Ni);

            for(int k=0; k<N; k++)
                pop[ ( a + come_island*Np ) * N + k ] = tmp_pop[ sorted[ id*Np ] * N + k ];
            p_fit[  a + come_island*Np  ] = p_fit[ sorted[ id*Np ] ];
        }
    }
}

__global__ void migration_ring_fps(double *pop, double *tmp_pop, double *p_fit, curandState * __restrict state, double MR, int Np, int Ni)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int island_id = x + y * blockDim.x * gridDim.x;
    double sum_fit = 0.f;

    int mr = Np / MR;
    if(mr==0)
        mr = 1;

    if(island_id < Ni)
    {
        for(int i = 0; i < Np; i++)
                sum_fit += p_fit[i + island_id*Np];

        for(int i = 0; i < mr; i++)
            for(int j = 0; j < Np; j++)
            {
                double offset = 0.f;
                double shut = curand_uniform_double(&state[island_id]);

                    offset += p_fit[j + island_id*Np] / sum_fit;
                    if(shut < offset)
                    {
                        int a =  curand(&state[island_id])%Np;
                        for(int k=0; k < N ; k++)
                            pop[  ((a + Np)%(Np*Ni)) * N + k ] = tmp_pop[(j + island_id*Np) * N + k];
                        p_fit[ ((a + Np)%(Np*Ni)) ] = p_fit[ j + island_id*Np ];
                        break;
                    }
            }
    }
}

double* createShiftVector(int dim, double min,double max) 
{
    double* d;
    double  hw, middle;
    double  s;
    int     i;
    hw     = (D(0.5) * (max - min));
    middle = (min + hw);
    d      = new double[dim];


    for (i = (dim - 1); i >= 0; i--) {

        do {
            double tempGauss = nextGaussian();
            s = (middle + (tempGauss * hw));
        } while ((s < min) || (s > max));
        d[i] = s;

    }
    return(d);
}

void bound_init(double *UB, double *LB, int BENCH)
{
    if(BENCH==1||BENCH==4||BENCH==7||BENCH==8||BENCH==9||BENCH==12||BENCH==13||BENCH==14||BENCH==15) { UB[0] = 100.0;  LB[0] = -100.0; }
    if(BENCH==2||BENCH==5||BENCH==10) { UB[0] = 5.0;    LB[0] = -5.0;   }
    if(BENCH==3||BENCH==6||BENCH==11) { UB[0] = 32.0;   LB[0] = -32.0;  }

}

void __global__ printt(double *pop)
{
    for(int i=0;i<4;i++)
        printf("%lf\n",pop[i]);
}

double* readOvectorVec(int dim)
{
  // read O vector from file in csv format, seperated by s_size groups
  //double* d = (double**) malloc(s_size*sizeof(double*));
    double * d;
    d = new double[2000];
  stringstream ss;
  ss<< "cdatafiles/"  << "F14"  << "-xopt.txt";
  ifstream file (ss.str());
  string value;
  string line;
  int c = 0;                      // index over 1 to dim
  int i = -1;                      // index over 1 to s_size
  int up = 0;                      // current upper bound for one group
int s[20] = {50,50,25,25,100,100,25,25,50,25,100,25,100,50,25,25,25,100,50,25};

  if (file.is_open())
    {
      stringstream iss;
      while ( getline(file, line) )
        {
          if (c==up)             // out (start) of one group
            {
              // printf("=\n");
              i++;
    //          d[i] =  (double*) malloc(s[i]*sizeof(double));
              up += s[i];
            }
          iss<<line;
          while (getline(iss, value, ','))
            {
              // printf("c=%d\ts=%d\ti=%d\tup=%d\tindex=%d\n",c,s[i],i,up,c-(up-s[i]));
              d[i*20+(c-(up-s[i]))] = stod(value);
              // printf("1\n");
              c++;
            }
          iss.clear();
          // printf("2\n");
        }
      file.close();
    }
  return d;
}

int main(int argc, char *argv[])
{
    MODEL = atoi(argv[1]);
    //SUB_MODEL = atoi(argv[2]);
    Np = atoi(argv[2]);
    Ni = atoi(argv[3]);
    MR = atoi(argv[4]);
    MF = atoi(argv[5]);
    DEVICE_ID = atoi(argv[6]);
    int BENCH = atoi(argv[7]);
/*
    Np = 20;
    if(MODEL==0||MODEL==1||MODEL==3)
    {
        if(SUB_MODEL==0)
            Ni = 128;
        if(SUB_MODEL==1)
            Ni = 256;
        if(SUB_MODEL==2)
            Ni = 512;
        if(SUB_MODEL==3)
            Ni = 1024;
        IL_DIM = sqrt(Ni);
    }
    if(MODEL==2)
    {
        if(SUB_MODEL==0)
            Ni = 132;
        if(SUB_MODEL==1)    
            Ni = 256;
        if(SUB_MODEL==2)
            Ni = 506;
        if(SUB_MODEL==3)
            Ni = 1024;  
        IL_DIM = sqrt(Ni);
        if(SUB_MODEL==0||SUB_MODEL==2)
            IL_DIM = 0;
    }
*/

    IL_DIM = sqrt(Ni);
    ANp = (int)round(Np * arc_rate);
    bound_init(&UB,&LB,BENCH);

    cudaSetDevice(DEVICE_ID);
    FILE *fp, *fp_island_div, *fp_div, *fp_island_fit, *fp_fit;
    double *var, *h_var;
    char filename[512];
    char filename_island_div[512];
    char filename_div[512];
    char filename_island_fit[512];
    char filename_fit[512];

    sprintf(filename,"DE_b2r_MODEL%d_20180807_N%d_Np%d_Ni%d_MF%d_MR%lf_T%lldBENCH%d.csv",MODEL,N,Np,Ni,MF,MR,FE,BENCH);
    sprintf(filename_island_div,"DE_island_div_MODEL%d_20180807_N%d_Np%d_Ni%d_MF%d_MR%lf_T%lldBENCH%d.csv",MODEL,N,Np,Ni,MF,MR,FE,BENCH);
    sprintf(filename_div,"DE_div_MODEL%d_20180807_N%d_Np%d_Ni%d_MF%d_MR%lf_T%lldBENCH%d.csv",MODEL,N,Np,Ni,MF,MR,FE,BENCH);
    sprintf(filename_island_fit,"DE_island_fit_MODEL%d_20180807_N%d_Np%d_Ni%d_MF%d_MR%lf_T%lldBENCH%d.csv",MODEL,N,Np,Ni,MF,MR,FE,BENCH);
    sprintf(filename_fit,"DE_fit_MODEL%d_20180807_N%d_Np%d_Ni%d_MF%d_MR%lf_T%lldBENCH%d.csv",MODEL,N,Np,Ni,MF,MR,FE,BENCH);

    fp     = fopen(filename,"w+");
    fp_island_div     = fopen(filename_island_div,"w+");
    fp_div            = fopen(filename_div,"w+");
    fp_island_fit     = fopen(filename_island_fit,"w+");
    fp_fit            = fopen(filename_fit,"w+");


    srand(time(NULL));
    curandState *devState;
    
    double *h_fit;

    double *d_pop,
           *d_p_fit,
           *d_mutation,
           *d_m_fit, *d_rand;

    double *sub_fit;
    int *key_out;    


    dim3 blockSize(32);
    long long int dim_gridSize = iDivUp(Np*Ni*N, 32);
    int pop_gridSize = iDivUp(Np*Ni, 32);
    int sub_pop_gridSize = iDivUp(Np*Ni*10, 32);

    int *offsets, *pop_seq;
    double *keys, *tmp_pop_fit;
    if(MODEL!=0)
    {
        gpuErrchk( cudaMalloc((void**)&pop_seq,  Np*Ni*sizeof(int)));
        gpuErrchk( cudaMalloc((void**)&offsets, (Ni+2)*sizeof(int)));
        gpuErrchk( cudaMalloc((void**)&keys,     Np*Ni*sizeof(double)));
        gpuErrchk( cudaMalloc((void**)&tmp_pop_fit,     Np*Ni*sizeof(double)));
    }

    h_fit = new double [Np*Ni];
    h_var = new double [N];

    gpuErrchk( cudaMalloc((void**)&var, N*sizeof(double)));
    gpuErrchk( cudaMalloc((void**)&key_out, Np*Ni*sizeof(int)));
    gpuErrchk( cudaMalloc((void**)&sub_fit, Np*Ni*10*sizeof(double)));
    gpuErrchk( cudaMalloc((void**)&d_pop, N*Np*Ni*sizeof(double)));
    gpuErrchk( cudaMalloc((void**)&d_mutation, N*Np*Ni*sizeof(double)));
    gpuErrchk( cudaMalloc((void**)&d_p_fit, Np*Ni*sizeof(double)));
    gpuErrchk( cudaMalloc((void**)&d_m_fit, Np*Ni*sizeof(double)));
    gpuErrchk( cudaMalloc((void**)&d_rand, Np*Ni*3*sizeof(double)));

    gpuErrchk( cudaMalloc((void**)&devState, N*Np*Ni*sizeof(curandState)));

    curand_setup_kernel<<<dim_gridSize,blockSize>>>(devState,time(NULL), Np, Ni);

    
    double *h_Ovector, *d_Ovector, *d_anotherz, *d_r25, *d_r50, *d_r100;
    int *h_Pvector, *d_Pvector;
    double *h_OvectorVec, *d_OvectorVec;

    gpuErrchk( cudaMalloc((void**)&d_anotherz, N*Np*Ni*sizeof(double)));

    gpuErrchk( cudaMalloc((void**)&d_OvectorVec, 1000*sizeof(double)));
    gpuErrchk( cudaMalloc((void**)&d_Pvector, 1000*sizeof(int)));
    gpuErrchk( cudaMalloc((void**)&d_Ovector, 1000*sizeof(double)));
    if(BENCH==13||BENCH==14)
        h_Pvector = createPermVector(905,LB,UB);
    else
        h_Pvector = createPermVector(1000,LB,UB);


    h_Ovector = createShiftVector(1000,LB,UB);
    cudaMemcpy       (d_Ovector, h_Ovector , 1000*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy       (d_Pvector, h_Pvector , 1000*sizeof(int), cudaMemcpyHostToDevice);

    h_OvectorVec = readOvectorVec(1000);
    cudaMemcpy       (d_OvectorVec, h_OvectorVec , 1000*sizeof(double), cudaMemcpyHostToDevice);

    double *h_r25;//[25*25];
    double *h_r50;//[50*50];
    double *h_r100;//[100*100];
    if(BENCH!=1&&BENCH!=2&&BENCH!=3&&BENCH!=12&&BENCH!=15)
    {
        gpuErrchk( cudaMalloc((void**)&d_r25, 25*25*sizeof(double)));
        gpuErrchk( cudaMalloc((void**)&d_r50, 50*50*sizeof(double)));
        gpuErrchk( cudaMalloc((void**)&d_r100, 100*100*sizeof(double)));

        h_r25 = readR(25,BENCH);
        h_r50 = readR(50,BENCH);
        h_r100 = readR(100,BENCH);

        cudaMemcpy   (d_r25, h_r25 , 25*25*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy   (d_r50, h_r50 , 50*50*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy   (d_r100, h_r100 , 100*100*sizeof(double), cudaMemcpyHostToDevice);
    }
    
    double hmin;
    int run_time = RUNTIME;
    int TIME = 0;
    long long int fit_eva = 0;

    clock_t start;
    double duration = 0;

    while(run_time--)
    {
        start = clock();
        init_pop <<<dim_gridSize, blockSize >>>(d_pop, devState, Np, Ni, LB, UB);
        if(MODEL!=0)
        {
            init_parameter  <<<pop_gridSize, blockSize>>>(pop_seq, Np, Ni);
            init_offsets <<<Ni+1,1>>>(offsets, Np, Ni);
        }

        TIME = fit_eva = 0;
        fitness_GPU      <<<pop_gridSize, blockSize>>>(d_pop, d_p_fit, Np, Ni, d_Ovector, d_anotherz, BENCH,d_Pvector,d_OvectorVec,d_r25,d_r50,d_r100);
        //assigntofit <<<pop_gridSize, blockSize>>>(d_p_fit, sub_fit, Np, Ni);
        cudaDeviceSynchronize();
        while(fit_eva < FE)
        {
            cudaMemcpy       (d_mutation, d_pop , Np*Ni*N*sizeof(double),cudaMemcpyDeviceToDevice);
            generate_rand    <<<pop_gridSize, blockSize>>> (d_rand, devState, Np, Ni);

            recombination    <<<dim_gridSize, blockSize>>>(d_pop, d_mutation, d_rand, devState, Np, Ni, UB, LB);
            fitness_GPU      <<<pop_gridSize, blockSize>>>(d_mutation, d_m_fit, Np, Ni, d_Ovector, d_anotherz, BENCH,d_Pvector,d_OvectorVec,d_r25,d_r50,d_r100);
            //cudaDeviceSynchronize();            
//assigntofit      <<<pop_gridSize, blockSize>>>(d_m_fit, sub_fit, Np, Ni);
            selection        <<<pop_gridSize, blockSize>>>(d_pop, d_p_fit, d_mutation, d_m_fit, Np, Ni);
            cudaMemcpy       (h_fit, d_p_fit , Np*Ni*sizeof(double),cudaMemcpyDeviceToHost);

            void *d_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
        
            if(MODEL!=0)
            {
            cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_p_fit, d_m_fit,
                    pop_seq, key_out, Np*Ni, Ni+1, offsets, offsets + 1);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_p_fit, d_m_fit,
                    pop_seq, key_out, Np*Ni, Ni+1, offsets, offsets + 1);
            cudaFree(d_temp_storage);
            }

            if(run_time < 5)
            {
                cudaMemcpy       (h_fit, d_p_fit , Np*Ni*sizeof(double), cudaMemcpyDeviceToHost);
////////////////////////ISLAND_DIV///////////////////////
                double var_sum = 0.f;
                for(int k = 0; k < Ni; k++)
                {
                    measure_div<<<N, 1>>>(d_pop, var, Np, Ni, N, k);
                    cudaMemcpy       (h_var, var , N*sizeof(double), cudaMemcpyDeviceToHost);

                    for(int m = 0; m < N; m++)
                        var_sum += h_var[m];
                }
                fprintf(fp_island_div,"%lf,", (var_sum / N) / Ni);
////////////////////////TOTAL_DIV///////////////////////
                double var_total_sum = 0.f;
                measure_div_total<<<N, 1>>>(d_pop, var, Np, Ni, N); 
                cudaMemcpy       (h_var, var , N*sizeof(double), cudaMemcpyDeviceToHost);
                for(int m = 0; m < N; m++)
                    var_total_sum += h_var[m];
                fprintf(fp_div,"%lf,", var_total_sum / N);
                
//////////////////////ISLAND_FIT/////////////////////////
                double island_avg_div = 0.f;
                for(int island_id = 0; island_id < Ni; island_id++)
                {
                    double island_fit_avg = 0.f;
                    double island_sum_div = 0.f;
 
                    for(int d = 0; d < Np; d++)
                        island_fit_avg += h_fit[d + island_id*Np];
                    island_fit_avg /= Np;
                    for(int d = 0; d < Np; d++)
                        island_sum_div += (h_fit[d + island_id*Np] - island_fit_avg) * (h_fit[d + island_id*Np] - island_fit_avg);
                    island_sum_div /= Np;
                    island_sum_div = sqrt(island_sum_div);
                    island_avg_div += island_sum_div;
                }
                fprintf(fp_island_fit,"%lf,", island_avg_div/Ni);

//////////////////////TOTAL FIT DIV//////////////////////
                double fit_avg = 0.f;
                double fit_div = 0.f;
                
                for(int d = 0; d < Np*Ni; d++)
                    fit_avg += h_fit[d];
                fit_avg /= Np*Ni;
                for(int d = 0; d < Np*Ni; d++)
                    fit_div += (h_fit[d] - fit_avg) * (h_fit[d] - fit_avg);
                fit_div /= Np*Ni;
                fit_div = sqrt(fit_div);
                fprintf(fp_fit,"%lf,", fit_div);
            }
            
            if(MODEL!=0)
                if(TIME%MF==0 && TIME!=0)
                {
                    cudaMemcpy       (d_mutation, d_pop , N*Np*Ni*sizeof(double), cudaMemcpyDeviceToDevice); //d_mutation for temp value

                    if(MODEL==1)
                        migration_ring_b2r      <<<pop_gridSize, blockSize>>>(d_pop, d_mutation, d_p_fit, key_out, devState, MR, Np, Ni);
                    if(MODEL==2)
                        migration_fully_b2r     <<<pop_gridSize, blockSize>>>(d_pop, d_mutation, d_p_fit, key_out, devState, MR, Np, Ni);
                    if(MODEL==3)
                        migration_island_b2r    <<<pop_gridSize, blockSize>>>(d_pop, d_mutation, d_p_fit, key_out, devState, Np, Ni, DIM, IL_DIM, MR);
                }
            
            cudaMemcpy       (h_fit, d_p_fit , Np*Ni*sizeof(double), cudaMemcpyDeviceToHost);

//printt<<<1,1>>>(d_p_fit);
//cudaDeviceSynchronize();
//exit(0);

            hmin = h_fit[0];
            for(int k = 0; k < Np*Ni; k++)
                if(hmin > h_fit[k])
                    hmin = h_fit[k];
            fprintf(fp,"%lf,",hmin);
//printf("hmin:%lf\n",hmin);
//printf("\n");
            TIME++;
            fit_eva += Ni*Np;

        }
        duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
        fprintf(fp,"\n");
        fprintf(fp_island_div,"\n");
        fprintf(fp_div,"\n");
        fprintf(fp_island_fit,"\n");
        fprintf(fp_fit,"\n");

        cout<<"times: "<< duration << " " << "generation"<<TIME<<" "<<"RUN_TIME"<<run_time<<" ";
        printf("DE_fps_MODEL%d_N%d_Np%d_Ni%d_MR%lf_MF%d_T%lldBENCH%d\n",MODEL,N,Np,Ni,MR,MF,FE,BENCH);
        fflush(stdout);
    }

    fclose(fp);
}

