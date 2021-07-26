
#include "prune.h"
#include <math.h>

float dynamic_epsilon = 0;
// float test_epsilon[PRUNE_TEST_LEN] = {0, 
// 0.001, 0.002, 0.004, 0.008, 
// 0.01, 0.02, 0.04, 0.08, 
// 0.1, 0.2, 0.4, 0.8, 
// 1, 2, 4, 8};
float test_epsilon[PRUNE_TEST_LEN] = {0, 0.01, 0.1, 1};

struct sat_one_layer_feature {
    long long param; //all of parameter
    long long prune_param; //the number of pruned channels
} solf;
int layer_cnt;

struct sat_feature {
    long long param; //all of parameter
    long long ch; //all of channel
    long long sparse_values;
    long long prune_ch; //the number of pruned channels
    long long prune_param; //the number of pruned channels
} sf;

struct avg_feature {
    double param, ch, sparse_values, prune_ch, prune_param;
} avg;

int *bitmap_prune = NULL;
int *bitmap_prune_gpu = NULL;


void prune_init() {
    bitmap_prune = (int *)calloc(BITMAP_PRUNE_SIZE, sizeof(int));
#ifdef GPU
    bitmap_prune_gpu = cuda_make_int_array(0, BITMAP_PRUNE_SIZE);
#endif
}

void prune_free() {
    free(bitmap_prune);
#ifdef GPU
    cuda_free(bitmap_prune_gpu);
#endif
}

//prune feature map
void prune_conv_feature(layer *l, network *net, int i_group) {
    int ch = l->c / l->groups;
    int area = l->h * l->w;
    int flag = 0, base = 0;
    float criterion = ((float)area * 0.99);
    for (int i = 0; i < ch; ++i) {
        flag = 0;
        base = i_group * ch * area + i * area;
        for (int j = 0; j < area; ++j) {
            if (fabs(net->input[base + j]) - dynamic_epsilon < 1e-6) flag++;
        }
        if (flag > criterion) {
            bitmap_prune[i] = 1;
        } else {
            bitmap_prune[i] = 0;
        }
#ifdef SAT_FEATURE
        if (bitmap_prune[i]) {
            sf.prune_ch++; // pruned channel
            sf.prune_param += area; // the number of values of pruned channels
            solf.prune_param += area; 
        }
        sf.sparse_values += flag;
#endif
    }
#ifdef SAT_FEATURE
    sf.param += ch * area;
    solf.param += ch * area;
    sf.ch += ch;
#endif
}

void prune_init_layer() {
#ifdef SAT_FEATURE
    solf.param = 0;
    solf.prune_param = 0;
#endif   
}

void prune_output_layer() {
#ifdef SAT_FEATURE
    // printf("\n****** Predict layer test result ******\n");
    // printf("layer feature (no, param, prune_param): %lld,%lld,%lld,%f,%lld,%lld\n", 
    // ++layer_cnt, solf.param, solf.prune_param, solf.prune_param * 1.0 / solf.param, solf.param * 4, solf.prune_param * 4);
    // printf("****** **************************** ******\n");
#endif
}

void prune_init_predict() {
#ifdef SAT_FEATURE
    sf.param = 0;
    sf.ch = 0;
    sf.sparse_values = 0;
    sf.prune_ch = 0;
    sf.prune_param = 0;
#endif
}

void prune_output_predict() {
#ifdef SAT_FEATURE
    avg.param += sf.param;
    avg.ch += sf.ch;
    avg.sparse_values += sf.sparse_values;
    avg.prune_ch += sf.prune_ch;
    avg.prune_param += sf.prune_param;

    printf("\n****** Predict Pruned test result ******\n");
    printf("feature: param, ch, sparse_values, prune_ch, prune_param\n%lld, %lld, %lld, %lld, %lld\n",
    sf.param, sf.ch, sf.sparse_values, sf.prune_ch, sf.prune_param);
    printf("****** **************************** ******\n");
#endif
}

void prune_init_valid() {
#ifdef SAT_FEATURE
    avg.param = 0;
    avg.ch = 0;
    avg.sparse_values = 0;
    avg.prune_ch = 0;
    avg.prune_param = 0;
    
    sf.param = 0;
    sf.ch = 0;
    sf.sparse_values = 0;
    sf.prune_ch = 0;
    sf.prune_param = 0;
#endif
}

void prune_output_valid(int img_num, float epsilon, float avg_time, float top1, float top5) {
#ifdef SAT_FEATURE
    printf("\n****** Valid Pruned test result ****** %d %lf\n", img_num);
    avg.param /= img_num;
    avg.ch /= img_num;
    avg.sparse_values /= img_num;
    avg.prune_ch /= img_num;
    avg.prune_param /= img_num;

    
    printf("AVG feature: param, ch, sparse_values, prune_ch, prune_param, epsilon, avg_time, top1, top5\nzeepres, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf\n",
    avg.param, avg.ch, avg.sparse_values, avg.prune_ch, avg.prune_param, epsilon, avg_time, top1, top5);
    printf("****** **************************** ******\n");
#endif

}