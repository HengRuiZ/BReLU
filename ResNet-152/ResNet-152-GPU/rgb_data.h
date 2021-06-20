/*
RGB data input and output
*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "mat.cuh"
typedef unsigned char datatype; //8 bit pixel
typedef long long labeltype; //int64 label
#define NUM_CLASSES 1000
#define NUM_IMAGES 50000
#define IMAGE_HEIGHT 224
#define IMAGE_WIDTH 224
#define IMAGE_CHANNEL 3

class rgb_data {
public:
	rgb_data(int batch_size, int height, int width, int channel);
	int next_batch(const char *imgfile, const char *labelfile);
	int read_frame(const char *imgfile, const char *labelfile, int n);
	int preprocess(void);
	int loadPred_GPU(void*v);
	double batch_accuracy(int topk);
	double accuracy(void);
	int save_pred_as(const char* filename);
	~rgb_data(void);

	int batch_size, h, w, channel, inSize;
	datatype *image;
	xwtype *ppro;
	labeltype *label;
	convtype *pred;//predictions of size batch_size*NUM_CLASSES
	int iter;//number of iterations
	double accuracy15[2];
	bool top1_true[NUM_IMAGES] = { false }, top5_true[NUM_IMAGES] = { false };
};
