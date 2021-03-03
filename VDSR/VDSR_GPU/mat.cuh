#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cudnn.h"

#define FRAME 1
#define CHANNEL 1
#define HEIGHT 240
#define WIDTH 416

//BLOCKSIZE和GRIDSIZE需要根据数据调节，暂时设置偏小以适应所有数据
#define BLOCKSIZE 1024
#define GRIDSIZE 1024
#if HEIGHT*WIDTH*16/BLOCKSIZE >= 2048
#define MAXGRID 1024
#elif HEIGHT*WIDTH*16/BLOCKSIZE >= 1024
#define MAXGRID 512
#elif HEIGHT*WIDTH*16/BLOCKSIZE >= 512
#define MAXGRID 256
#endif

//#define INT8_EXT_CONFIG
//#define INT8x4_EXT_CONFIG
#define FLOAT_CONFIG

#ifdef INT8_EXT_CONFIG
#define XWFORMAT CUDNN_TENSOR_NHWC
#define XWTYPE CUDNN_DATA_INT8
#define YFORMAT CUDNN_TENSOR_NHWC
#define YTYPE CUDNN_DATA_FLOAT
#define BFORMAT CUDNN_TENSOR_NHWC
#define BTYPE CUDNN_DATA_INT32
#define CONVTYPE CUDNN_DATA_INT32
#define ALGO CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
typedef char xwtype;
typedef float convtype;//data type after convolution, bias and activation
typedef int btype;
typedef int midtype;//data type converted from uvtype
#define THRESHOLD 127
#endif
#ifdef INT8x4_EXT_CONFIG
#define XWFORMAT CUDNN_TENSOR_NCHW_VECT_C// Input and output features maps must be multiple of 4 
#define XWTYPE CUDNN_DATA_INT8x4
#define YFORMAT CUDNN_TENSOR_NCHW
#define YTYPE CUDNN_DATA_FLOAT
#define BFORMAT CUDNN_TENSOR_NCHW
#define BTYPE CUDNN_DATA_FLOAT
#define CONVTYPE CUDNN_DATA_INT32
#define ALGO CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
typedef char xwtype;
typedef float convtype;//data type after convolution, bias and activation
typedef int midtype;//data type converted from uvtype
#define THRESHOLD 127
#endif
#ifdef FLOAT_CONFIG
#define XWFORMAT CUDNN_TENSOR_NCHW// Input and output features maps must be multiple of 4 
#define XWTYPE CUDNN_DATA_FLOAT
#define YFORMAT CUDNN_TENSOR_NCHW
#define YTYPE CUDNN_DATA_FLOAT
#define BFORMAT CUDNN_TENSOR_NCHW
#define BTYPE CUDNN_DATA_FLOAT
#define CONVTYPE CUDNN_DATA_FLOAT
#define ALGO CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
typedef float xwtype;
typedef float convtype;//data type after convolution, bias and activation
#define THRESHOLD 127
#endif

__global__ void findMax_reduce1(convtype *g_idata, convtype *g_odata, int n);//n>=GRIDSIZE*BLOCKSIZE*2
__global__ void findMax_reduce2(convtype *data);//number = 1024
__host__ void findMax(convtype *data, convtype *buffer, int n, convtype *max);
//__global__ void conv2mid(convtype *conv, midtype *mid, int num);
//__global__ void VectorDiv(midtype *dividend, xwtype *quotient, int divisor, int n);
__global__ void applyRes(unsigned char *in, xwtype *res, unsigned char *recon);
char* HWCN2NCHW_VECT_C_CPU(char *HWCN, int H, int W, int C, int N, int *outSize);
char* NCHW2NCHW_VECT_C_CPU(char *NCHW, int N, int C, int H, int W, int *outSize);
char* NCWH2HWCN_CPU(char *NCWH, int H, int W, int C, int N, int *outSize);
xwtype* HWCN2NCHW_CPU(xwtype*HWCN, int H, int W, int C, int N, int *outSize);
__global__ void CHW2CHW_VECT_C_QUANT(convtype *dividend, xwtype *quotient, int channelSize, int channel, int gridSize, int multiplier, int shifts);//应明确指定用途，尤其是正负数
int NCHW2NCHW_VECT_C_QUANT(convtype *before, xwtype *after, int N, int C, int H, int W, int multiplier, int shifts);
__global__ void CHW2CHW_VECT_C_QUANT_BLU(convtype *input, xwtype *output, int channelSize, int channel, int gridSize, int blu, int multiplier, int shifts);//只用于同时有BLU和QUANT的情况
int NCHW2NCHW_VECT_C_QUANT_BLU(convtype *before, xwtype *after, int N, int C, int H, int W, int blu, int multiplier, int shifts);//同上
//void HWCN2NCHW_VECT_C(char *HWCN, char *NCHW, int H, int W, int C, int N);//C<=4 and expand C to 4
