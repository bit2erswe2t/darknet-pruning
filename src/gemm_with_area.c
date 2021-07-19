#include "gemm_with_area.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "prune.h"

void gemm_nn_with_area(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc,
        int area)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
#ifdef PRUNE
            if (bitmap_prune[k / area]) { // if current channel can be pruned
                k += area - 1; // jump to the next channel
                continue;
            }
#endif
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt_with_area(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc,
        int area)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn_with_area(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc,
        int area)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt_with_area(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc,
        int area)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_cpu_with_area(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc,
        int area)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn_with_area(M, N, K, ALPHA,A,lda, B, ldb,C,ldc, area);
    else if(TA && !TB)
        gemm_tn_with_area(M, N, K, ALPHA,A,lda, B, ldb,C,ldc, area);
    else if(!TA && TB)
        gemm_nt_with_area(M, N, K, ALPHA,A,lda, B, ldb,C,ldc, area);
    else
        gemm_tt_with_area(M, N, K, ALPHA,A,lda, B, ldb,C,ldc, area);
}

void gemm_with_area(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc,
        int area)
{
    gemm_cpu_with_area( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc, area);
}