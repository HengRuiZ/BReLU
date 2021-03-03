#include "vdsr.cuh"

VDSR::VDSR(cudnnHandle_t cudnnHandle, int batch, int inHeight, int inWidth, int inChannel, int layers)
{
	int i;
	this->batch = batch;
	this->inHeight = inHeight;
	this->inWidth = inWidth;
	this->inChannel = inChannel;
	this->layers = layers;
	I1.build(batch, inChannel, inHeight, inWidth);
	C_in.build(cudnnHandle, I1.xDesc, batch, I1.height, I1.width, I1.channel, 64, 3, 1);
	C_layers = new CovLayer[layers];
	C_layers[0].build(cudnnHandle, C_in.yDesc, batch, inHeight, inWidth, 64, 64, 3, 1);
	for(i=1;i<layers;i++)
		C_layers[i].build(cudnnHandle, C_layers[i-1].yDesc, batch, inHeight, inWidth, 64, 64, 3, 1);
	C_out.build(cudnnHandle, C_layers[layers-1].yDesc, batch, inHeight, inWidth, 64, 1, 3, 1);
	workspaceSize = C_in.workspaceSize;
	for(i=0;i<layers;i++)
		workspaceSize = workspaceSize > C_layers[i].workspaceSize ? workspaceSize : C_layers[i].workspaceSize;
	workspaceSize = workspaceSize > C_out.workspaceSize ? workspaceSize : C_out.workspaceSize;
	cudaMalloc(&workspace, workspaceSize);
}
int VDSR::load_data(xwtype*input)
{
	cudaDeviceSynchronize();
	I1.load(input);
	return 0;
}
#ifdef INT8x4_EXT_CONFIG
int VDSR::load_para(const char*fn)
{
	int i;
	FILE *fp;
	if (fopen_s(&fp, fn, "rb"))
	{
		printf("open file %s failed\n", fn);
		exit(1);
	}
	C_in.load_para(fp);
	for (i = 0;i < layers;i++)
		C_layers[i].load_para(fp);
	C_out.load_para(fp);
	fseek(fp, -4, SEEK_END);
	fread(&ratio_out, sizeof(float), 1, fp);
	fclose(fp);
	return 0;
}
int VDSR::forward(cudnnHandle_t cudnnHandle)
{
	int i;
	C_in.ConvForward(cudnnHandle, I1.xDesc, I1.x, workspace);
	C_in.activate(cudnnHandle);
	C_in.quantize_out();
	//C_in.viewmem((xwtype*)I1.x);
	C_layers[0].ConvForward(cudnnHandle, C_in.yDesc, C_in.y, workspace);
	C_layers[0].activate(cudnnHandle);
	C_layers[0].quantize_out();
	//C_layers[0].viewmem((xwtype*)C_in.y);
	for (i = 1;i < layers;i++)
	{
		C_layers[i].ConvForward(cudnnHandle, C_layers[i - 1].yDesc, C_layers[i - 1].y, workspace);
		C_layers[i].activate(cudnnHandle);
		C_layers[i].quantize_out();
		//C_layers[i].viewmem((xwtype*)C_layers[i - 1].y);
	}
	C_out.ConvForward(cudnnHandle, C_layers[layers - 1].yDesc, C_layers[layers - 1].y, workspace);
	//C_out.viewmem((xwtype*)C_layers[layers - 1].y);
	return 0;
}
#elif defined FLOAT_CONFIG
int VDSR::load_para(const char*fn)
{
	int i;
	FILE *fp;
	if (fopen_s(&fp, fn, "rb"))
	{
		printf("open file %s failed\n", fn);
		exit(1);
	}
	C_in.load_para(fp);
	for (i = 0;i < layers;i++)
		C_layers[i].load_para(fp);
	C_out.load_para(fp);
	fclose(fp);
	return 0;
}
int VDSR::forward(cudnnHandle_t cudnnHandle)
{
	int i;
	C_in.ConvForward(cudnnHandle, I1.xDesc, I1.x, workspace);
	C_in.activate(cudnnHandle);
	//C_in.viewmem((xwtype*)I1.x);
	C_layers[0].ConvForward(cudnnHandle, C_in.yDesc, C_in.y, workspace);
	C_layers[0].activate(cudnnHandle);
	//C_layers[0].viewmem((xwtype*)C_in.y);
	for (i = 1;i < layers;i++)
	{
		C_layers[i].ConvForward(cudnnHandle, C_layers[i - 1].yDesc, C_layers[i - 1].y, workspace);
		C_layers[i].activate(cudnnHandle);
		//C_layers[i].viewmem((xwtype*)C_layers[i - 1].y);
	}
	C_out.ConvForward(cudnnHandle, C_layers[layers - 1].yDesc, C_layers[layers - 1].y, workspace);
	//C_out.viewmem((xwtype*)C_layers[layers - 1].y);
	return 0;
}
int VDSR::quantizeNsave(const char*STEP_FILE, const char*BLU_FILE, const char*QUANT_MODEL)
{
	int i;
	FILE *f_step, *f_blu, *f_quant;
	float temp;
	if (fopen_s(&f_step, STEP_FILE, "rb"))
	{
		printf("open file %s failed\n", STEP_FILE);
		exit(1);
	}
	if (fopen_s(&f_blu, BLU_FILE, "rb"))
	{
		printf("open file %s failed\n", BLU_FILE);
		exit(1);
	}
	if (fopen_s(&f_quant, QUANT_MODEL, "wb"))
	{
		printf("open file %s failed\n", QUANT_MODEL);
		exit(1);
	}
	C_in.quantizeNsave(f_step, f_blu, f_quant);
	for (i = 0;i < layers;i++)
		C_layers[i].quantizeNsave(f_step, f_blu, f_quant);
	C_out.quantizeNsave(f_step, f_blu, f_quant);
	fread(&temp, sizeof(float), 1, f_blu);//ratio_out
	fwrite(&temp, sizeof(float), 1, f_quant);//ratio_out
	fclose(f_step);
	fclose(f_blu);
	fclose(f_quant);
	return 0;
}
#endif
VDSR::~VDSR(void)
{
	delete[] C_layers;
	cudaFree(workspace);
	return;
}
