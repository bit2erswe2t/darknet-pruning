#ifndef __PRUNE_H__
#define __PRUNE_H__

#include "darknet.h"

/*---------------  zeep ----------------*/
#define BITMAP_PRUNE_SIZE 66536
#define PRUNE_TEST_LEN 4

extern int *bitmap_prune;
extern int *bitmap_prune_gpu;
extern float dynamic_epsilon; 
extern float test_epsilon[PRUNE_TEST_LEN];

void prune_conv_feature(layer *l, network *net, int i_group);
void prune_init_predict();
void prune_init_valid();
void prune_output_predict();
void prune_output_valid(int, float, float, float, float);
void prune_init();
void prune_free();

/*---------------  zeep ----------------*/
#endif //__PRUNE_H__