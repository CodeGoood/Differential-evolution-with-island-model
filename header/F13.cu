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
#include <thrust/device_ptr.h>
#include <limits.h>
#include <math.h>
#include <cub/cub.cuh>

#include <sstream>
#include <vector>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include "para.cuh"
#include "F1.cuh"
#include "F2.cuh"
#include "F3.cuh"
#include "F4.cuh"
#include "F7.cuh"

int dimension = 905;


__host__ __device__ double  rotateVectorConform(int i, int *c, double *anotherz1, int *s, double *anotherz, int *Pvector, double *r25, double *r50, double *r100)
{
    double z[100];
  
  // copy values into the new vector
    int overlap = 5;
    for (int j = c[0] - i*overlap; j < c[0] +s[i] - i*overlap; ++j)
    {
        double op = anotherz[Pvector[j]];
        z[j-(c[0] - i*overlap)] = op;
    }
  // cout<<"copy done"<<endl;
    if (s[i]==25)
    {
      multiply( z, r25, s[i],anotherz1);
    }
    else if (s[i] == 50)
    {    
      multiply( z, r50, s[i],anotherz1);
    }
    else if (s[i] == 100) 
    {
      multiply( z, r100, s[i],anotherz1);
    }
  
    c[0] = c[0] + s[i];
}

__host__ __device__ double F13(double *x, double *Ovector, double *anotherz, int *Pvector, double *r25, double *r50, double *r100)
{
    int i;
    double result=0.0;

    double anotherz1[100];

    int s[20] = {50,50,25,25,100,100,25,25,50,25,100,25,100,50,25,25,25,100,50,25};

    double w[20];
    w[0] = 0.4353328319185867;
    w[1] = 0.009916326715665044;
    w[2] = 0.05427408462402158;
    w[3] = 29.36277114669338;
    w[4] = 11490.33037696204;
    w[5] = 24.12830828465555;
    w[6] = 3.451118275947202;
    w[7] = 2.326453155544301;
    w[8] = 0.001766789509539211;
    w[9] = 0.02539477160659147;
    w[10] = 19.9959220973553;
    w[11] = 0.0003668001927378567;
    w[12] = 0.001356048928320048;
    w[13] = 0.03874911849895608;
    w[14] = 88.89452353634552;
    w[15] = 57901.31382337087;
    w[16] = 0.008485316099078568;
    w[17] = 0.07362038148350014;
    w[18] = 0.688309295752457;
    w[19] = 119314.8936123031;
 
    for(i=0;i<905;i++)
    {
        anotherz[i]=x[i]-Ovector[i];
    }
    int* c = new int[1];
    c[0] = 0;

    for (i = 0; i < 20; i++)
    {
        rotateVectorConform(i, c, anotherz1, s, anotherz, Pvector, r25, r50, r100);
        //anotherz1 = rotateVectorConform(i, c);
        result += w[i] * schwefel(anotherz1, s[i]);
    }
    delete[] c;
    return  result;
}
