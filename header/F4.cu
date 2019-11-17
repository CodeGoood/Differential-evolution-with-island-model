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
using namespace std;

int64_t m_seed= L(0);//functionInitRandomSeed;
int64_t M  = 0x5DEECE66D;
int64_t A  = 0xB;
int64_t MASK = ((L(1)) << (L(48))) - (L(1));
double m_nextGaussian;
bool m_havenextGaussian = false;


int next(int bits) 
{
    int64_t s;
    int64_t result;
    m_seed = s = (((m_seed * M) + A) & MASK);
    result = (s >> (L(48 - bits)));
    return((int)result);
}

int nextInt(int n) {
  int bits, val;

  if ((n & (-n)) == n) {
    return((int) ((n * L(next(31))) >> L(31)));
  }

  do {
    bits = next(31);
    val  = bits % n;
  } while (bits - val + (n - 1) < 0);

  return(val);
}

double nextDouble()
{
    return ((((L(next(26))) <<(L(27))) + (L(next(27)))) / (double) ((L(1)) << (L(53))));
}

double nextGaussian()
{
    double multiplier, v1, v2, s;

    if (m_havenextGaussian) {
        m_havenextGaussian = false;
        return(m_nextGaussian) ;
    }

    do {
        v1 = ((D(2.0) * nextDouble()) - D(1.0));
        v2 = ((D(2.0) * nextDouble()) - D(1.0));
        s  = ((v1 * v1) + (v2 * v2));
    } while ((s >= D(1.0)) || (s <= D(0.0)));

    multiplier = sqrt(D(-2.0) * log(s) / s);
    m_nextGaussian    = (v2 * multiplier);
    m_havenextGaussian = true;
    return (v1 * multiplier);
}

int* createPermVector(int dim, double min,double max){
  int* d;
  int  i, j, k, t;
  //d = (int*)malloc(sizeof(int) * dim);
  d = new int[dim];

  for (i = (dim - 1); i >= 0; i--) {
    d[i] = i;
  }

  for (i = (dim << 3); i >= 0; i--) {
    j = nextInt(dim);

    do {
      k = nextInt(dim);
    } while (k == j);

    t    = d[j];
    d[j] = d[k];
    d[k] = t;
  }

  return(d);
}

double* readR(int sub_dim, int BENCH)
{
    double *m = new double[sub_dim*sub_dim];
    stringstream ss;
    ss<< "cdatafiles/" << "F" << BENCH << "-R"<<sub_dim<<".txt";
    // cout<<ss.str()<<endl;

    ifstream file (ss.str());
    string value;
    string line;
    int i=0;
    int j;

    if (file.is_open())
    {
        stringstream iss;
        while ( getline(file, line) )
        {
            j=0;
            iss<<line;
            while (getline(iss, value, ','))
            {
                m[i * sub_dim + j] = stod(value);
                j++;
            }
            iss.clear();
            i++;
        }
        file.close();
    }
    else
    {
        cout<<"Cannot open datafiles "<< BENCH <<endl;
    }
    return m;
}

__host__ __device__ void multiply(double *vector, double *matrix, int dim, double *anotherz1)
{
    int    i,j;
    //double*result = (double*)malloc(sizeof(double) * dim);

    for(i = dim - 1; i >= 0; i--) {
        anotherz1[i] = 0;

        for(j = dim - 1; j >= 0; j--) {
            anotherz1[i] += vector[j] * matrix[i * dim + j];
        }
    }

}

__host__ __device__ void rotateVector(int i, int *c, double *anotherz1, int *s, double *anotherz, int *Pvector, double *r25, double *r50, double *r100)
{
    //double* z = (double*) malloc(sizeof(double)*s[i]);
    //double* z = new double[s[i]];
    double z[100];
    for (int j = c[0]; j < c[0]+s[i]; ++j)
    {
        // cout<<"j-c "<<j-c<<" p "<<Pvector[j]<<endl;
        z[j-c[0]] = anotherz[Pvector[j]];
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
    else
    {
        printf("size of rotation matrix out of range\n");
    }
    //delete []z;
    c[0] = c[0] + s[i];
}

__host__ __device__ double F4(double *x, double *Ovector, double *anotherz, int *Pvector, double *r25, double *r50, double *r100)
{
     int    i;
    double result=0.0;
    //double anotherz[1000];
    double anotherz1[100];
    int s[7];
    s[0] = 50;
    s[1] = 25;
    s[2] = 25;
    s[3] = 100;
    s[4] = 50;
    s[5] = 25;
    s[6] = 25;

    double w[7];
    w[0] = 45.69963061477328;
    w[1] = 1.564615888932325;
    w[2] = 18465.32344576193;
    w[3] = 0.01108949891829191;
    w[4] = 13.62598489888553;
    w[5] = 0.3015150617722511;
    w[6] = 59.6078373100912;


    for(i = 0; i < N; i++) {
        anotherz[i] = x[i] - Ovector[i];
    }

    int* c = new int[1];
    c[0] = 0;
    for (i = 0; i < 7; i++)
    {
        rotateVector(i, c, anotherz1, s, anotherz, Pvector, r25, r50, r100);
        result += w[i] * elliptic(anotherz1, s[i]);
        //delete []anotherz1;
    }
    //double* z = new double[dimension-c];
    double z[700];
    //double* z = (double*) malloc(sizeof(double)*(N-c));
    //double* z = new double [N-c[0]];
    for (i = 300; i < N; i++)
        z[i-300] = anotherz[Pvector[i]];

    result += elliptic(z, 700);
    delete[] c;
    //return(result);
    return  result; 
}
