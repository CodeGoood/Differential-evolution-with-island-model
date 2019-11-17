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


__host__ __device__ double   rotateVectorConflict(int i, int *c, double *x, double *anotherz1, int *s, double *anotherz, int *Pvector, double *OvectorVec, double *r25, double *r50, double *r100)
{
    double z[100];
    int overlap = 5;
  // printf("i=%d\tc=%d\tl=%d\tu=%d\n", i, c, c - (i)*overlap, c +s[i] - (i)*overlap);
  
  // copy values into the new vector
  for (int j = c[0] - i*overlap; j < c[0] +s[i] - i*overlap; ++j)
    {
      // cout<<"j-c "<<j-c<<" p "<<Pvector[j]<<endl;
      z[j-(c[0] - i*overlap)] = x[Pvector[j]] - OvectorVec[i*20+(j-(c[0] - i*overlap))];
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

__host__ __device__ double F14(double *x, double *OvectorVec, double *anotherz, int *Pvector, double *r25, double *r50, double *r100)
{
    int i;
    double result=0.0;

    double anotherz1[100];

    int s[20] = {50,50,25,25,100,100,25,25,50,25,100,25,100,50,25,25,25,100,50,25};

    double w[20];
    w[0] = 0.4753902976291743;
w[1] = 498729.4349012464;
w[2] = 328.1032851136715;
w[3] = 0.3231599525017396;
w[4] = 136.4562810151285;
w[5] = 9.025518074644136;
w[6] = 0.092439618796505;
w[7] = 0.0001099484417540517;
w[8] = 0.009366186525282616;
w[9] = 299.6790409975172;
w[10] =4.939589864702469;
w[11] =81.36413957775696;
w[12] =0.6544027477491213;
w[13] =11.61197332124502;
w[14] =2860774.320110003;
w[15] =8.583578119678344e-05;
w[16] =23.56951753728851;
w[17] =0.04810314216448019;
w[18] =1.4318494811719;
w[19] =12.16976123256558;

    int* c = new int[1];
    c[0] = 0;

    for (i = 0; i < 20; i++)
    {
        rotateVectorConflict(i, c, x, anotherz1, s, anotherz, Pvector, OvectorVec, r25, r50, r100);
        //anotherz1 = rotateVectorConform(i, c);
        result += w[i] * schwefel(anotherz1, s[i]);
    }
    delete[] c;
    return  result;
}
