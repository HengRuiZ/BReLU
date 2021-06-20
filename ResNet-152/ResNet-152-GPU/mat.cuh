#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cudnn.h"

//#define ORI_FILE "..\\..\\data\\HEVC_Sequence\\Traffic_2560x1600_30_crop.yuv"
//#define INPUT_FILE "..\\..\\data\\anchor16.0\\Traffic_intra_main_HM16.0_anchor_Q%d.yuv"
//#define ORI_FILE "..\\..\\data\\HEVC_Sequence\\PeopleOnStreet_3840x2160_30_420_08_150_crop10.yuv"
//#define INPUT_FILE "..\\..\\data\\anchor16.0\\PeopleOnStreet_intra_main_HM16.0_anchor_"
//#define ORI_FILE "..\\..\\data\\HEVC_Sequence\\BQTerrace_1920x1080_60_10.yuv"
//#define INPUT_FILE "..\\..\\data\\anchor16.0\\BQTerrace_intra_main_HM16.0_anchor_Q%d.yuv"
//#define ORI_FILE "..\\..\\data\\HEVC_Sequence\\Johnny_1280x720_60_crop10.yuv"
//#define INPUT_FILE "..\\..\\data\\anchor16.0\\Johnny_intra_main_HM16.0_anchor_Q%d.yuv"
//#define ORI_FILE "..\\..\\data\\HEVC_Sequence\\BQMall_832x480_60_crop10.yuv"
//#define INPUT_FILE "..\\..\\data\\anchor16.0\\BQMall_intra_main_HM16.0_anchor_"
#define ORI_FILE "..\\..\\data\\HEVC_Sequence\\BasketballPass_416x240_50_crop10.yuv"
#define INPUT_FILE "..\\..\\data\\anchor16.0\\BasketballPass_intra_main_HM16.0_anchor_"
//#define ORI_FILE "..\\..\\data\\HEVC_Sequence\\BQSquare_416x240_60.yuv"
//#define INPUT_FILE "..\\..\\data\\anchor16.0\\BQSquare_intra_main_HM16.0_anchor_Q%d.yuv"
//#define ORI_FILE "..\\..\\data\\HEVC_Sequence\\BlowingBubbles_416x240_50.yuv"
//#define INPUT_FILE "..\\..\\data\\anchor16.0\\BlowingBubbles_intra_main_HM16.0_anchor_"
//#define ORI_FILE "..\\..\\data\\HEVC_Sequence\\RaceHorses_416x240_30_crop10.yuv"
//#define INPUT_FILE "..\\..\\data\\anchor16.0\\RaceHorses_intra_main_HM16.0_anchor_Q%d.yuv"
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
//#define UINT8x4_EXT_CONFIG
#define INT8x4_EXT_CONFIG
//#define FLOAT_CONFIG

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
typedef float btype;
typedef int midtype;//data type converted from uvtype
#define THRESHOLD 127
#endif
#ifdef UINT8x4_EXT_CONFIG//in doubt
#define XWTYPE CUDNN_DATA_UINT8x4
#define CONVTYPE CUDNN_DATA_INT32
#define YTYPE CUDNN_DATA_FLOAT
#define XWFORMAT CUDNN_TENSOR_NHWC// Input and output features maps must be multiple of 4 
#define YFORMAT CUDNN_TENSOR_NHWC
#define BFORMAT CUDNN_TENSOR_NHWC
#define BTYPE CUDNN_DATA_FLOAT
#define ALGO CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
typedef unsigned char xwtype;
typedef float convtype;//data type after convolution, bias and activation
typedef float btype;
typedef int midtype;//data type converted from uvtype
#define THRESHOLD 128
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
typedef float btype;
#define THRESHOLD 128
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
xwtype* HWCN2NHWC4_CPU(xwtype*HWCN, int H, int W, int C, int N, int *outSize);
__global__ void CHW2CHW_VECT_C(convtype *dividend, xwtype *quotient, int divisor, int channelSize, int channel, int gridSize);
int NCHW2NCHW_VECT_C(convtype *dividend, xwtype *quotient, int divisor, int N, int C, int H, int W);
//void HWCN2NCHW_VECT_C(char *HWCN, char *NCHW, int H, int W, int C, int N);//C<=4 and expand C to 4
__global__ void mul_shift(convtype *input, xwtype *output, int inSize, int multiplier, int shifts);
__global__ void mul_shift_inplace(convtype *input, int inSize, int multiplier, int shifts);
__global__ void CHW2CHW_VECT_C_QUANT_BLU(convtype *input, xwtype *output, int channelSize, int channel, int gridSize, int blu, int multiplier, int shifts);//只用于同时有BLU和QUANT的情况
int NCHW2NCHW_VECT_C_QUANT_BLU(convtype *before, xwtype *after, int N, int C, int H, int W, int blu, int multiplier, int shifts);//同上
double mse(float*x, int size, int gpu, const char*fn, int offset);
int load_tensor(float*x, int size, const char*fn, int offset);
