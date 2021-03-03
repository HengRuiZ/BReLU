#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include "cnn.cuh"

//#define MEM_DBG

class BasicBlock {
public:
	BasicBlock(void);
	int build(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int batch,int inHeight,int inWidth,int inChannel,int outChannel);
	int load_para(FILE*fp);
	int quantizeNsave(FILE *f_step, FILE *f_blu, FILE *f_quant);
	int forward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void*x, void*workspace);
	~BasicBlock(void);

	int batch;
	int inHeight, inWidth, inChannel;
	int outHeight, outWidth, outChannel;
	int ksize = 3;
	int stride; // for the 1st layer and expansion for the last
	int downsample;// 0 for straight and 1 for downsample
	int mul_d, shift_d;

	CovLayer C1, C2, Cd;//conv1, conv2, downsample(optional)
	int workspaceSize;
};
class BottleneckBlock {
public:
	BottleneckBlock(void);
	int build(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int batch, int inHeight, int inWidth, int inChannel, int inChannel_expasion, int outChannel);
	int load_para(FILE*fp);
	int forward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void*x, void*workspace);
	~BottleneckBlock(void);

	int batch;
	int inHeight, inWidth, inChannel;
	int outHeight, outWidth, outChannel;
	int ksize = 3;//3 for C2 and 1 for C1 and C3 and Cd
	int stride;
	int expansion = 4;
	int downsample;
	CovLayer C1, C2, C3, Cd;
	int workspaceSize;
};
class BlockLayer {
public:
	BlockLayer(void);
	cudnnTensorDescriptor_t build(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, int batch, int inHeight, int inWidth, int inChannel, int inChannel_expansion, int outChannel, int block_class, int block_num);
	int load_para(FILE*fp);
	int quantizeNsave(FILE *f_step, FILE *f_blu, FILE *f_quant);
	void* forward(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, void*x, void*workspace);
	~BlockLayer(void);

	int block_class, block_num;
	int expansion;
	int batch;
	int stride;
	int inHeight, inWidth, inChannel;
	int outHeight, outWidth, outChannel;
	int workspaceSize;
	BasicBlock*B_Blocks;
	BottleneckBlock*BN_Blocks;
};
class Resnet{
public:
	Resnet(cudnnHandle_t cudnnHandle, int batch, int inHeight, int inWidth, int inChannel, int blockclass, int*blocks);
	int load_para(const char*fn);
	int load_data(xwtype*input);
	int forward(cudnnHandle_t cudnnHandle);
	int quantizeNsave(const char*STEP_FILE, const char*BLU_FILE, const char*QUANT_MODEL);
	~Resnet(void);

	int batch;
	int inHeight = 224, inWidth = 224, inChannel = 3;
	int block_num[4];
	int layers, blockclass;//0 for plain while 1 for bottleneck
	InputLayer I1;//input layer
	CovLayer C1;//first conv layer
	PoolLayer P1;//first pooling layer
	BlockLayer B1, B2, B3, B4;//residual blocks
	CovLayer FC1;//fully connected layer at the end
	PoolLayer P2;
	cudnnTensorDescriptor_t outDesc[4];
	void*out;
	int workspaceSize;
	void*workspace;
};

