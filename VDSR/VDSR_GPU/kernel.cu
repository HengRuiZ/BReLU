#include <sstream>
#include <stdlib.h>
#include <windows.h>
#include <time.h>
#include "vdsr.cuh"
#include "yuv_data.h"

#define DATA_HEAD "data\\%dx.data"

#define MODEL_F_NAME "model\\model.data"
#define MODEL_Q_NAME "model\\model_q.data"
#define QUANT_DATA "model\\quant.data"
#define BLU_DATA "model\\blu.data"
#ifdef INT8x4_EXT_CONFIG
#define MODEL_NAME MODEL_Q_NAME
#elif defined(FLOAT_CONFIG)
#define MODEL_NAME MODEL_F_NAME
void quantizeNsave(void)//only for FLOAT_CONFIG
{
	int num_gpus;
	cudnnHandle_t cudnnHandle;
	cudaGetDeviceCount(&num_gpus);
	cudaSetDevice(0);
	cudnnCreate(&cudnnHandle);
	VDSR vdsr(cudnnHandle, 1, 512, 512, 1, 18);
	vdsr.load_para(MODEL_F_NAME);
	vdsr.quantizeNsave(QUANT_DATA, BLU_DATA, MODEL_Q_NAME);
}
#endif // INT8x4_EXT_CONFIG

void run_frame(FrameData*frame_data,cudnnHandle_t cudnnHandle)
{
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;
	VDSR vdsr(cudnnHandle, 1, frame_data->h, frame_data->w, frame_data->outChannel, 18);
	vdsr.load_para(MODEL_NAME);
	vdsr.load_data(frame_data->ppro);
	QueryPerformanceFrequency(&Frequency);

	cudaDeviceSynchronize();
	QueryPerformanceCounter(&StartingTime);
	vdsr.forward(cudnnHandle);
	cudaDeviceSynchronize();
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	printf("time:%lldus\n", ElapsedMicroseconds.QuadPart);
	frame_data->loadRes_GPU((convtype*)vdsr.C_out.u, vdsr.ratio_out);
	frame_data->applyRes();
	frame_data->count_psnr();
	char savename[100];
	sprintf(savename, "%dx%dx%d.raw", frame_data->w, frame_data->h, frame_data->scale);
	frame_data->save_recon_as(savename);
	return;
}
/*
{
	int i;
	int num_gpus;
	vrcnn_data test_data(frame, height, width);
	double psnr1, psnr2;
	time_t now;
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;
	FILE*logfile;

	cudaGetDeviceCount(&num_gpus);
	qvrcnn qvrcnn1(0, 1, channel, height, width);//GPU_num,NCHW
	qvrcnn1.load_para(model_fn);
	test_data.read_data(ori_fn, input_fn);
	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	for (i = 0;i < frame;i++)
	{
		qvrcnn1.load_data(test_data.input + i*channel*height*width);
		qvrcnn1.forward();
		cudaDeviceSynchronize();
		cudaMemcpy(test_data.recon + i*channel*height*width, (datatype*)qvrcnn1.I1.x_rec, channel*height*width, cudaMemcpyDeviceToHost);
	}
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	now = time(0);
	//test_data.save_recon_as("recon.yuv");
	//test_data.psnr_pf();
	psnr1 = test_data.psnr(test_data.input);
	psnr2 = test_data.psnr(test_data.recon);
	printf("\nbefore net:PSNR=%f\nafter quantized net:PSNR=%f\ntime:%lldus\n", psnr1, psnr2, ElapsedMicroseconds.QuadPart);
	if (fopen_s(&logfile, "log.txt", "a+"))
		printf("write file failed\n");
	fprintf(logfile, "\nQVRCNN test date:%sdata:%s\nframes:%d\nheight:%d\nwidth:%d\nbefore net:PSNR=%f\nafter quantized net:PSNR=%f\ntime:%lldus\n", ctime(&now), input_fn, frame, height, width, psnr1, psnr2, ElapsedMicroseconds.QuadPart);
	fclose(logfile);
}*/
int run_batch(const char*data_fn,int scale, cudnnHandle_t cudnnHandle)
{
	int i;
	yuv_data batch(data_fn, scale);
	for (i = 0;i < batch.frame;i++)
		run_frame(&batch.data[i], cudnnHandle);
	batch.count_psnr();
	printf("scale:%d\n", scale);
	printf("psnr_input:%.2f,psnr_recon:%.2f\n", batch.psnr_input, batch.psnr_recon);
	return 0;
}
int run_all(void)
{
	int scale;
	char data_fn[200];
	int num_gpus;
	cudnnHandle_t cudnnHandle;
	cudaGetDeviceCount(&num_gpus);
	cudaSetDevice(0);
	cudnnCreate(&cudnnHandle);
	for (scale = 2; scale < 5; scale++)
	{
		sprintf_s(data_fn, DATA_HEAD, scale);
		run_batch(data_fn, scale, cudnnHandle);
	}
	cudnnDestroy(cudnnHandle);
	return 0;
}
int main(int argc, char**argv)
{
	//quantizeNsave();//only for FLOAT_CONFIG
	run_all();
	//run_all(argv[1],argv[2],atoi(argv[3]),atoi(argv[4]));
	//system("pause");
}
