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

__host__ __device__ double F8(double *x, double *Ovector, double *anotherz, int *Pvector, double *r25, double *r50, double *r100)
{
     int    i;
    double result=0.0;
    //double anotherz[1000];
    double anotherz1[100];

    int s[20] = {50,50,25,25,100,100,25,25,50,25,100,25,100,50,25,25,25,100,50,25};

    double w[20];
    w[0] = 4.630303898328862;
    w[1] = 0.6864279131323788;
    w[2] = 1143756360.088768;
    w[3] = 2.007758000992542;
    w[4] = 789.3671746186714;
    w[5] = 16.33320168691342;
    w[6] = 6.074996976773932;
    w[7] = 0.06466850615348552;
    w[8] = 0.07569676542592878;
    w[9] = 35.67259952037679;
    w[10] = 7.972550897120909e-06;
    w[11] = 10.78226361560055;
    w[12] = 4.199965822662956e-06;
    w[13] = 0.001923872337445647;
    w[14] = 0.001677064168931893;
    w[15] = 686.7975656834401;
    w[16] = 0.1571465735122922;
    w[17] = 0.04417781474359776;
    w[18] = 0.3543888360330344;
    w[19] = 0.006051787005326174;

    for(i = 0; i < N; i++) {
        anotherz[i] = x[i] - Ovector[i];
    }

    int* c = new int[1];
    c[0] = 0;
    for (i = 0; i < 20; i++)
    {
        rotateVector(i, c, anotherz1, s, anotherz, Pvector, r25, r50, r100);
        result += w[i] * elliptic(anotherz1, s[i]);
        //delete []anotherz1;
    }
    delete[] c;
    //return(result);
    return  result;
}









