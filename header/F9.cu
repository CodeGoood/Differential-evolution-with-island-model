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

__host__ __device__ double F9(double *x, double *Ovector, double *anotherz, int *Pvector, double *r25, double *r50, double *r100)
{
     int    i;
    double result=0.0;
    //double anotherz[1000];
    double anotherz1[100];

    int s[20] = {50,50,25,25,100,100,25,25,50,25,100,25,100,50,25,25,25,100,50,25};

    double w[20];
    w[0] = 1756.996997333392;
    w[1] = 570.7338357630722;
    w[2] = 3.355970801461051;
    w[3] = 1.036411046690371;
    w[4] = 62822.29234664552;
    w[5] = 1.731558445257169;
    w[6] = 0.08980493862761418;
    w[7] = 0.0008071244581233227;
    w[8] = 1403745.636331398;
    w[9] = 8716.208361607407;
    w[10] = 0.003344616275362139;
    w[11] = 1.34951151390475;
    w[12] = 0.004776798216929033;
    w[13] = 5089.913308957042;
    w[14] = 12.66641964611824;
    w[15] = 0.0003588940639560592;
    w[16] = 0.2400156872122657;
    w[17] = 3.964353127212945;
    w[18] = 0.001428559155593465;
    w[19] = 0.005228218198001427;

    for(i = 0; i < N; i++) {
        anotherz[i] = x[i] - Ovector[i];
    }

    int* c = new int[1];
    c[0] = 0;
    for (i = 0; i < 20; i++)
    {
        rotateVector(i, c, anotherz1, s, anotherz, Pvector, r25, r50, r100);
        result += w[i] * rastrigin(anotherz1, s[i]);
        //delete []anotherz1;
    }
    delete[] c;
//    return(result);
    return result;
}

