#include <cudnn.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mat.cuh"

#define check(status) do{									\
	if(status!=CUDNN_STATUS_SUCCESS)											\
		{													\
			printf("cudnn returned none 0.\n");				\
			cudaDeviceReset();                              \
			exit(1);										\
		}													\
}while(0)

class CovLayer {
public:
	CovLayer();
	int build(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int batch, int height, int width, int inChannel, int outChannel, int ksize,int stride);//build layer
	int load_para(FILE *fp);//copy paras from file to GPU memory
	int quantizeNsave(FILE *step, FILE *blu,FILE *quant, float ratio, int fc);
	int ConvForward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void *x, void *workspace, size_t workspaceSize);
	int applyBias(cudnnHandle_t cudnnHandle);
	int batch_norm(cudnnHandle_t cudnnHandle);
	int applyRes(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t resDesc, void*res);
	int activate(cudnnHandle_t cudnnHandle);
	int quantize_out(void);//u->y,blu,mul,shift
	int quantize_u(int mul, int shift);
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
	size_t workspaceSize, memspace;//workspace size needed and GPU memory consumed
	float alpha = 1;
	float beta = 0;
	// convolution without bias
	cudnnConvolutionFwdAlgo_t AlgoPerf;
	cudnnConvolutionFwdPreference_t ConvPerf = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
	cudnnConvolutionDescriptor_t convDesc;//¾í»ýÃèÊö·û
	cudnnFilterDescriptor_t wDesc;//kernel
	void *w;
	cudnnTensorDescriptor_t uDesc;//output descriptor
	void *u;
	// batch norm
	cudnnBatchNormMode_t bnMode=CUDNN_BATCHNORM_SPATIAL;
	cudnnTensorDescriptor_t yDesc, MeanVarScaleBiasDesc;
	void *y, *mean, *var, *scale, *bias;
	double epsilon = 1e-5;
	// activation
	int blu;
	int mul, shift;
	cudnnActivationDescriptor_t actiDesc;
	//int algo_num;//number of avaliable algorithms
	//cudnnConvolutionFwdAlgoPerf_t perfResults[8];//¾í»ýËã·¨ÃèÊö·û
	int builded = 0;
};
class InputLayer {
public:
	InputLayer(void);
	int build(int batch, int channel, int height, int width);
	int load(xwtype *input);
	int ppro(cudnnHandle_t cudnnHandle);
	int viewmem(xwtype*res);
	~InputLayer(void);
	
	int batch;
	int inHeight, inWidth, inChannel;
	int outHeight, outWidth, outChannel;
	int inSize, outSize;
	float alpha = 1, beta = 0;
	cudnnTensorDescriptor_t xDesc, x_outDesc;
	void *x,*x_out;
};
class PoolLayer {
public:
	PoolLayer(void);
	int build(cudnnPoolingMode_t mode, int batch, int channel, int inHeight, int inWidth, int ksize, int stride);
	int pool(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t inDesc, void*in);
	int quantizeNsave(FILE *blu, FILE *quant);
	int load_blu(FILE*fp);
	int quantize_out(void);
	int quantize_u(int mul, int shift);
	int viewmem(void);
	~PoolLayer(void);

	int batch;
	int channel;
	int inHeight, inWidth;
	int outHeight, outWidth;
	int outSize;
	int ksize;
	int stride;
	int padding;
	cudnnPoolingMode_t mode; //CUDNN_POOLING_MAX CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
	float alpha = 1, beta = 0;
	int blu;
	int mul, shift;
	cudnnPoolingDescriptor_t poolingDesc;
	cudnnTensorDescriptor_t uDesc,yDesc;
	void*u,*y;
};
