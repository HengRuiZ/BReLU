/*
YUV data input and output
*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "mat.cuh"

#define MAX_FRAME 100
typedef unsigned char datatype; //此处适用于8bit像素位宽
class FrameData
{
public:
	FrameData(void);
	int load(FILE*fp);
	int preprocess(void);
	int loadRes_GPU(convtype*v, float ratio_out);
	int applyRes(void);
	double count_psnr(void);
	int save_recon_as(char* filename);
	~FrameData(void);

	int build = 0;
	int scale;
	float ratio_out;
	int h, w, inChannel, outChannel, inSize, outSize;
	double psnr_input, psnr_recon;
	datatype*hr, *sr;
	float *bic;
	xwtype *ppro;
	convtype *res;
};
class yuv_data {
public:
	yuv_data(const char*fn, int scale);
	double count_psnr(void);
	~yuv_data(void);

	int frame, current_frame, scale;
	double psnr_input, psnr_recon;
	FrameData data[MAX_FRAME];
};
