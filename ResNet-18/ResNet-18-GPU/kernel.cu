#include <sstream>
#include <stdlib.h>
#include <windows.h>
#include <time.h>
#include "rgb_data.h"
#include "resnet.cuh"

#ifdef INT8x4_EXT_CONFIG
#define IMAGE_FILE "D:\\zhaohengrui\\data\\imagenet\\val_image_NCHW_VECT_C.data"
#elif defined FLOAT_CONFIG
#define IMAGE_FILE "D:\\zhaohengrui\\data\\imagenet\\val_image_NCHW.data"
#endif
#define LABEL_FILE "D:\\zhaohengrui\\data\\imagenet\\val_label.data"

#define RESNET152

#ifdef RESNET18
#ifdef INT8x4_EXT_CONFIG
#define RESNET_MODEL "model\\resnet18_blu35103_q.data"
#elif defined FLOAT_CONFIG
#define RESNET_MODEL "model\\resnet18_blu35103.data"
#endif
#define STEP_FILE "model\\quant_param89.data"
#define BLU_FILE "model\\3sigma.blu"
#define QUANT_FILE "model\\resnet18_blu35103_q.data"
#define BLOCKCLASS 0
#define BLOCK_NUM {2,2,2,2}
#elif defined RESNET50
#define RESNET_MODEL "model\\resnet50.data"
#define BLOCKCLASS 1
#define BLOCK_NUM {3,4,6,3}
#elif defined RESNET152
#ifdef INT8x4_EXT_CONFIG
#define RESNET_MODEL "model\\resnet152_blu100_q.data"
#elif defined FLOAT_CONFIG
#define RESNET_MODEL "model\\resnet152_blu100.data"
#endif
#define STEP_FILE "model\\resnet152_quant_param.data"
#define BLU_FILE "model\\resnet152_blu.data"
#define QUANT_FILE "model\\resnet152_blu100_q.data"
#define BLOCKCLASS 1
#define BLOCK_NUM {3,8,36,3}
#endif
int quantizeNsave(void)
{
	int i;
	int num_gpus;
	cudnnHandle_t cudnnHandle;
	cudaGetDeviceCount(&num_gpus);
	cudaSetDevice(0);
	cudnnCreate(&cudnnHandle);
	int resnet_blocks[4] = BLOCK_NUM;
	Resnet resnet(cudnnHandle, 1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL, BLOCKCLASS, resnet_blocks);
	resnet.load_para(RESNET_MODEL);
	resnet.quantizeNsave(STEP_FILE, BLU_FILE, QUANT_FILE);
	return 0;
}
int test_data(void)
{
	rgb_data val(50, 224, 224, 3);
	val.read_frame(IMAGE_FILE, LABEL_FILE, 12);
	val.next_batch(IMAGE_FILE, LABEL_FILE);
	val.preprocess();
	return 0;
}
int test_conv(void)
{
	int num_gpus;
	cudnnHandle_t cudnnHandle;
	rgb_data val(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL);
	InputLayer I1;
	CovLayer C1;
	PoolLayer P1;
	BasicBlock B1;
	BlockLayer BL1;
	FILE*para_fp;
	int workspaceSize;
	void*workspace;

	cudaGetDeviceCount(&num_gpus);
	cudaSetDevice(0);
	cudnnCreate(&cudnnHandle);
	I1.build(2, 128, 28, 28);
	C1.build(cudnnHandle, I1.x_outDesc, 2, 28, 28, 128, 128, 3, 1);
	P1.build(CUDNN_POOLING_MAX, 1, 64, C1.outHeight, C1.outWidth, 3, 2);
	B1.build(cudnnHandle, P1.uDesc, 1, P1.outHeight, P1.outWidth, 64, 64);
	BL1.build(cudnnHandle, P1.uDesc, 1, P1.outHeight, P1.outWidth, 64, 1, 64, 0, 2);
	workspaceSize = C1.workspaceSize > B1.workspaceSize ? C1.workspaceSize : B1.workspaceSize;
	cudaMalloc(&workspace, workspaceSize);

	load_tensor((float*)I1.x_out, 2 * 28 * 28 * 128, "block2.data", 0);
	load_tensor((float*)C1.w, 3 * 3 * 128 * 128, "block2.data", 2 * 28 * 28 * 128 * 2 * 4);
	C1.ConvForward(cudnnHandle, I1.x_outDesc, I1.x_out, workspace, workspaceSize);
	mse((float*)C1.u, 2 * 28 * 28 * 128, 1, "block2.data", 2 * 28 * 28 * 128 * 4);

	fopen_s(&para_fp, RESNET_MODEL, "rb");
	C1.load_para(para_fp);
	B1.load_para(para_fp);
	fclose(para_fp);
	val.next_batch(IMAGE_FILE, LABEL_FILE);
	val.preprocess();
	I1.load(val.ppro);
	//I1.ppro(cudnnHandle);
	C1.ConvForward(cudnnHandle, I1.xDesc, I1.x, workspace, C1.workspaceSize);
	C1.batch_norm(cudnnHandle);
	C1.activate(cudnnHandle);
	//C1.viewmem((xwtype*)I1.x);
	P1.pool(cudnnHandle, C1.yDesc, C1.y);
	//P1.viewmem();
	B1.forward(cudnnHandle, &P1, workspace);
	return 0;
}
int test_resnet(int batch_size)
{
	int i;
	int num_gpus;
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;
	cudnnHandle_t cudnnHandle;
	QueryPerformanceFrequency(&Frequency);
	cudaGetDeviceCount(&num_gpus);
	cudaSetDevice(0);
	cudnnCreate(&cudnnHandle);
	int resnet_blocks[4] = BLOCK_NUM;
	rgb_data val(batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL);
	Resnet resnet(cudnnHandle, batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL, BLOCKCLASS,resnet_blocks);
	resnet.load_para(RESNET_MODEL);
	//val.iter = 999;
	for (i = 0;i < NUM_IMAGES/batch_size;i++)
	{
		val.next_batch(IMAGE_FILE, LABEL_FILE);
		val.preprocess();//normalize
		resnet.load_data(val.ppro);

		cudaDeviceSynchronize();
		QueryPerformanceCounter(&StartingTime);
		resnet.forward(cudnnHandle);
		cudaDeviceSynchronize();
		QueryPerformanceCounter(&EndingTime);

		ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
		ElapsedMicroseconds.QuadPart *= 1000000;
		ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
		printf("time:%lldus\n", ElapsedMicroseconds.QuadPart);
		
		val.loadPred_GPU(resnet.FC1.u);
		val.batch_accuracy(5);
	}
	val.accuracy();
	return 0;
}
int run_all(void)
{
	//quantizeNsave();
	//test_data();
	//test_conv();
	test_resnet(50);
	return 0;
}
int main(int argc, char**argv)
{
	//run_all(ORI_FILE, INPUT_FILE, HEIGHT, WIDTH);
	run_all();
	//quantizeNsave();
	//system("pause");
}
