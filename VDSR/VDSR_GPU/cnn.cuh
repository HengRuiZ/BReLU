#include <cudnn.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mat.cuh"

#define check(status) do{									\
	if(status!=0)											\
		{													\
			printf("cudnn returned none 0.\n");				\
			cudaDeviceReset();                              \
			exit(1);										\
		}													\
}while(0)

class CovLayer {
public:
	CovLayer();
	int build(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int batch, int height, int width, int inChannel, int outChannel, int ksize, int stride);//build layer
	int load_para(FILE *fp);//copy paras from file to GPU memory
	int quantizeNsave(FILE *step, FILE *blu, FILE *quant);
	int ConvForward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void *x, void *workspace);
	int activate(cudnnHandle_t cudnnHandle);
	int quantize_out(void);//u->y,blu,mul,shift
	int viewmem(xwtype*x);
	int freeMem(void);//free data u and y, typically in reference
	int setMem(void);//set u and y
	~CovLayer();

	int batch;
	int inHeight, inWidth, inChannel;
	int outHeight, outWidth, outChannel;
	int ksize;
	int stride;
	int wSize, uySize;
	size_t workspaceSize;//workspace size needed and GPU memory consumed
	float alpha = 1;
	float beta = 0;
	// convolution with bias
	cudnnConvolutionDescriptor_t convDesc;//¾í»ýÃèÊö·û
	cudnnFilterDescriptor_t wDesc;//kernel
	void *w;
	cudnnTensorDescriptor_t uDesc, bDesc;//output descriptor
	void *u, *b;
	// activation
	cudnnActivationDescriptor_t actiDesc;
	// output
	int blu;
	int mul, shift;
	cudnnTensorDescriptor_t yDesc;
	void *y;
	int builded = 0;
};
class InputLayer {
public:
	InputLayer(void);
	int build(int batch, int channel, int height, int width);
	int load(xwtype *input);
	int viewmem(xwtype*res);
	~InputLayer(void);
	
	int batch;
	int height;
	int width;
	int channel;
	int size;
	cudnnTensorDescriptor_t xDesc;
	void *x;
};
