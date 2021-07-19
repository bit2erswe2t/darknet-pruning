
#include "prune.h"
#include <math.h>

float dynamic_epsilon = 0;
// float test_epsilon[PRUNE_TEST_LEN] = {0, 
// 0.001, 0.002, 0.004, 0.008, 
// 0.01, 0.02, 0.04, 0.08, 
// 0.1, 0.2, 0.4, 0.8, 
// 1, 2, 4, 8};
float test_epsilon[PRUNE_TEST_LEN] = {0,
0.01, 0.1, 1};

struct sat_feature {
    int param; //all of parameter
    int zero; //the number of zero
    int ch; //all of channel
    int zero_ch; //the number of the whole zero channel
    int prune_ch; //the number of pruned channels
    int prune_param; //the number of pruned channels
} sf;

struct avg_feature {
    float param, zero, ch, zero_ch, prune_ch, prune_param;
} avg;

int *bitmap_prune = NULL;
int *bitmap_prune_gpu = NULL;

//prune feature map
void prune_conv_feature(layer *l, network *net, int i_group) {
    int ch = l->c / l->groups;
    int area = l->h * l->w;
    int flag = 0, base = 0, flag2 = 0;
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
            sf.prune_param += flag; // the number of pruned channels
        }
        flag2 = 0;
        for (int j = 0; j < area; ++j) {
            if (fabs(net->input[base + j]) < 1e-6) flag2++;
        }
        if (flag2 > criterion) {
            sf.zero_ch++; // a zero channel 
        }
        sf.zero += flag; // the number of less than epsilon
#endif
    }
#ifdef SAT_FEATURE
    sf.param += ch * area;
    sf.ch += ch;
#endif
}

void prune_init_predict() {
#ifdef SAT_FEATURE
    avg.param += sf.param;
    avg.zero += sf.zero;
    avg.ch += sf.ch;
    avg.zero_ch += sf.zero_ch;
    avg.prune_ch += sf.prune_ch;
    avg.prune_param += sf.prune_param;

    sf.param = 0;
    sf.zero = 0;
    sf.ch = 0;
    sf.zero_ch = 0;
    sf.prune_ch = 0;
    sf.prune_param = 0;
#endif
}

void prune_init_valid() {
#ifdef SAT_FEATURE
    avg.param = 0;
    avg.zero = 0;
    avg.ch = 0;
    avg.zero_ch = 0;
    avg.prune_ch = 0;
    avg.prune_param = 0;
    
    sf.param = 0;
    sf.zero = 0;
    sf.ch = 0;
    sf.zero_ch = 0;
    sf.prune_ch = 0;
    sf.prune_param = 0;
#endif
}

void prune_output_predict() {
#ifdef SAT_FEATURE
    printf("\n****** Predict Pruned test result ******\n");
    printf("feature: param, zero, ch, zero_ch, prune_ch, prune_param\n%d, %d, %d, %d, %d, %d\n",
    sf.param, sf.zero, sf.ch, sf.zero_ch, sf.prune_ch, sf.prune_param);
    printf("****** **************************** ******\n");
#endif
}

void prune_output_valid(int img_num, float epsilon, float avg_time, float top1, float top5) {
#ifdef SAT_FEATURE
    printf("\n****** Valid Pruned test result ****** %d %f\n", img_num, avg.zero);
    avg.param /= img_num;
    avg.zero /= img_num;
    avg.ch /= img_num;
    avg.zero_ch /= img_num;
    avg.prune_ch /= img_num;
    avg.prune_param /= img_num;

    
    printf("AVG feature: param, zero, ch, zero_ch, prune_ch, prune_param, epsilon, avg_time, top1, top5\nzeepres, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n",
    avg.param, avg.zero, avg.ch, avg.zero_ch, avg.prune_ch, avg.prune_param, epsilon, avg_time, top1, top5);
    printf("****** **************************** ******\n");
#endif

}

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

// // prune conv weight
// void prune_conv_weight_gpu(layer *l) {
//     int size_num = l->n;
//     int size_ch = l->c / l->groups;
//     int size_area = l->size * l->size;
//     int size_cube = size_ch * size_area;
//     cuda_pull_array(l->weights_gpu, l->weights, size_num * size_cube);
//     all_w_param += size_num * size_cube;
//     for (int i = 0; i < size_ch; ++i) {
//         int flag = 1;
//         for (int j = 0; j < size_num; ++j){
//             for (int k = 0; k < size_area; ++k) {
//                 pruned_w_param += (fabs(l->weights[i * size_area + j * size_cube + k] - 1e-8) < dynamic_epsilon);
//                 flag = flag & (fabs(l->weights[i * size_area + j * size_cube + k] - 1e-8) < dynamic_epsilon);
//                 compare_cost += 1;
//                 if (!flag) break;
//             }
//             if (!flag) break;
//         }
            
//         bitmap_zp[i] |= flag;
//         if (flag) {
//             pruned_w_param_ch += size_num * size_area;
//             for (int j = 0; j < size_num; ++j)
//                 for (int k = 0; k < size_area && flag == 1; ++k)
//                     l->weights[i * size_area + j * size_cube + k] = 0;
//         }
//     }
//     all_im2col_load += size_ch * size_area * l->out_w * l->out_h;
//     for (int i = 0; i < size_ch; ++i) {
//         if (bitmap_zp[i])
//             saved_im2col_load += size_area * l->out_w * l->out_h;
//     }
//     cuda_push_array(l->weights_gpu, l->weights, size_num * size_cube);
// }

// //prune feature map
// void prune_conv_feature_gpu(layer *l, network *net) {
//     int size_ch = l->c / l->groups;
//     int size_area = l->h * l->w;
//     cuda_pull_array(net->input_gpu, net->input, size_ch * size_area);
//     all_f_param += size_ch * size_area;
//     store_cost += size_ch;
//     for (int i = 0; i < size_ch; ++i) {
//         int flag = 1;
//         for (int j = 0; j < size_area; ++j) {
//             pruned_f_param += (fabs(net->input[i * size_area + j] - 1e-8) < dynamic_epsilon);
//             flag = flag & (fabs(net->input[i * size_area + j] - 1e-8) < dynamic_epsilon);
//             compare_cost += 1;
//             if (!flag) break;
//         }
//         bitmap_zp[i] = flag;
//         if (flag) {
//             pruned_f_param_ch += size_area;
//             for (int j = 0; j < size_area; ++j)
//                 net->input[i * size_area + j] = 0;
//         }
//     }
//     cuda_push_array(net->input_gpu, net->input, size_ch * size_area);
// }

// //prune conv output 
// void prune_conv_output_gpu(layer *l) {
//     // printf("zeep test: %f\n", dynamic_epsilon);
//     int ch = l->n;
//     int area = l->out_h * l->out_w;
//     cuda_pull_array(l->output_gpu, l->output, l->outputs * l->batch);
//     all_f_param += l->outputs;
//     for (int i = 0; i < ch; ++i) {
//         int flag = 1;
//         for (int j = 0; j < area; ++j){
//             flag = flag & (fabs(l->output[i * area + j] - 1e-8) < dynamic_epsilon);
//         }
//         bitmap_zp[i] = flag;
//         if (flag) {
//             pruned_f_param_ch += area;
//             for (int j = 0; j < area; ++j)
//                 l->output[i * area + j] = 0;
//         }
//     }
//     cuda_push_array(l->output_gpu, l->output, l->outputs * l->batch);
// }

// // prune connnect weight
// void prune_fc_weight_gpu(layer *l) {
//     //return 0;
//     int in_size = l->inputs;
//     int out_size = l->outputs;
//     int size = in_size * out_size;
//     cuda_pull_array(l->weights_gpu, l->weights, size);
//     all_w_param += size;
//     store_cost += out_size;
//     for (int i = 0; i < out_size; ++i) {
//         int flag = 1;
//         for (int j = 0; j < in_size; ++j) {
//             pruned_w_param += (fabs(l->weights[i * in_size + j] - 1e-8) < dynamic_epsilon);
//             flag = flag & (fabs(l->weights[i * in_size + j] - 1e-8) < dynamic_epsilon);
//             compare_cost += 1;
// //            if (!flag) break;
//         }
//         bitmap_zp[i] = flag;
//         if (flag) {
//             for (int j = 0; j < in_size; ++j) {
//                 l->weights[i * in_size + j] = 0;
//             }
//         }
//     }
//     all_im2col_load += in_size * 2 * out_size;
//     for (int i = 0; i < out_size; ++i) {
//         if (bitmap_zp[i])
//             saved_im2col_load += in_size * 2;
//     }
//     cuda_push_array(l->weights_gpu, l->weights, size);
// }

// //prune feature map
// void prune_fc_feature_gpu(layer *l, network *net) {
//     // int in_size = l->inputs;
//     // cuda_pull_array(net->input_gpu, net->input, in_size);
//     // all_f_param += in_size;
//     // for (int i = 0; i < in_size; ++i) {
//     //     pruned_f_param += (fabs(net->input[i] - 1e-8) < dynamic_epsilon);
//     //     // if (fabs(net->input[i] - 1e-8) < dynamic_epsilon) {
//     //     //     net->input[i] = 0;
//     //     // }
//     // }
//     // cuda_push_array(net->input_gpu, net->input, in_size);
// }


// void prune_init() {
//     all_w_param = 0;
//     pruned_w_param_ch = 0;
//     all_f_param = 0;
//     pruned_f_param_ch = 0;
//     saved_im2col_load = 0;
//     all_im2col_load = 0;
//     pruned_w_param = 0;
//     pruned_f_param = 0;
//     compare_cost = 0;
//     store_cost = 0;
// }

// void prune_init_avg() {
//     pruned_w_param_ch_avg = 0;
//     pruned_f_param_ch_avg = 0;
//     saved_im2col_load_avg = 0;
//     pruned_w_param_avg = 0;
//     pruned_f_param_avg = 0;
//     compare_cost_avg = 0;
//     store_cost_avg = 0;
// }

// void prune_cal_avg(int m) {
//     pruned_w_param_ch_avg += pruned_w_param_ch * 1.0 / m;
//     pruned_f_param_ch_avg += pruned_f_param_ch * 1.0 / m;
//     saved_im2col_load_avg += saved_im2col_load * 1.0 / m;
//     pruned_w_param_avg += pruned_w_param * 1.0 / m;
//     pruned_f_param_avg += pruned_f_param * 1.0 / m;
//     compare_cost_avg += compare_cost * 1.0 / m;
//     store_cost_avg += store_cost * 1.0 / m;
// }

// void prune_output_avg() {
//     printf("\n****** Pruned test result ******\n");
//     printf("dynamic_epsilon: %f\n", dynamic_epsilon);
//     printf("all_w_param: %lld, pruned_w_param_ch_avg: %f, pruned_w_rate_ch: %f, pruned_w_param_avg: %f, pruned_w_rate: %f\n", all_w_param,
//                     pruned_w_param_ch_avg,
//                     pruned_w_param_ch_avg * 1.0 / all_w_param,
//                     pruned_w_param_avg,
//                     pruned_w_param_avg * 1.0 / all_w_param);
//     printf("all_f_param: %lld, pruned_f_param_ch_avg: %f, pruned_f_rate_ch: %f, pruned_f_param_avg: %f, pruned_f_rate: %f\n", all_f_param,
//                     pruned_f_param_ch_avg,
//                     pruned_f_param_ch_avg * 1.0 / all_f_param,
//                     pruned_f_param_avg,
//                     pruned_f_param_avg * 1.0 / all_f_param);
//     long long all_param = all_w_param + all_f_param;
//     float pruned_param_ch_avg = pruned_w_param_ch_avg + pruned_f_param_ch_avg;
//     float pruned_param_avg = pruned_w_param_avg + pruned_f_param_avg;
//     printf("all_param: %lld, pruned_param_ch_avg: %f, pruned_param_rate_ch: %f, pruned_param_avg: %f, pruned_param_rate: %f\n", all_param,
//                     pruned_param_ch_avg,
//                     pruned_param_ch_avg * 1.0 / all_param,
//                     pruned_param_avg,
//                     pruned_param_avg * 1.0 / all_param);
//     printf("all_im2col_load: %lld, saved_im2col_load_avg: %f, pruned_im2col_rate: %f\n", all_im2col_load,
//                     saved_im2col_load_avg,
//                     saved_im2col_load_avg * 1.0 / all_im2col_load);
//     printf("compare_cost_avg: %f, store_cost_avg:%f\n", compare_cost_avg, store_cost_avg);
//     printf("*****************\n");
// }

// void prune_output() {
//     printf("\n****** Pruned test result ******\ndynamic_epsilon: %f\nall_w_param: %lld, pruned_w_param_ch: %lld, pruned_w_rate_ch: %f, pruned_w_param: %lld, pruned_w_rate: %f\nall_f_param: %lld, pruned_f_param_ch: %lld, pruned_f_rate_ch: %f, pruned_f_param: %lld, pruned_f_rate: %f\nall_param: %lld, pruned_param_ch: %lld, pruned_param_rate_ch: %f, pruned_param: %lld, pruned_param_rate: %f\nall_im2col_load: %lld, saved_im2col_load: %lld, pruned_im2col_rate: %f\ncompare_cost: %lld, store_cost: %lld\n*****************\n\n", 
//                     dynamic_epsilon,
//                     all_w_param,
//                     pruned_w_param_ch,
//                     pruned_w_param_ch * 1.0 / all_w_param,
//                     pruned_w_param,
//                     pruned_w_param * 1.0 / all_w_param,
//                     all_f_param,
//                     pruned_f_param_ch,
//                     pruned_f_param_ch * 1.0 / all_f_param,
//                     pruned_f_param,
//                     pruned_f_param * 1.0 / all_f_param,
//                     all_w_param + all_f_param,
//                     pruned_w_param_ch + pruned_f_param_ch,
//                     (pruned_w_param_ch + pruned_f_param_ch) * 1.0 / (all_w_param + all_f_param),
//                     pruned_w_param + pruned_f_param,
//                     (pruned_w_param + pruned_f_param) * 1.0 / (all_w_param + all_f_param),
//                     all_im2col_load,
//                     saved_im2col_load,
//                     saved_im2col_load * 1.0 / all_im2col_load,
//                     compare_cost, store_cost);
// }