#ifndef GEMM_WIHT_AREA_H
#define GEMM_WIHT_AREA_H

void gemm_with_area(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc,
        int area);

#endif
