#include "yuv_data.h"

FrameData::FrameData(void)
{
	return;
}
int FrameData::load(FILE*fp)
{
	inChannel = 1;
	if (fread(&h, sizeof(int), 1, fp) == 0)
		return 0;
	if(fread(&w, sizeof(int), 1, fp)==0)
		return 0;
	inSize = w*h;
	if (XWFORMAT == CUDNN_TENSOR_NCHW_VECT_C)
		outChannel = 4;
	else
		outChannel = 1;
	outSize = w*h*outChannel;
	hr = (datatype*)malloc(sizeof(datatype)*inSize);
	sr = (datatype*)malloc(sizeof(datatype)*inSize);
	bic = (float*)malloc(sizeof(float)*inSize);
	res = (convtype*)malloc(sizeof(convtype)*inSize);
	ppro = (xwtype*)malloc(sizeof(xwtype)*outSize);
	fread(hr, sizeof(datatype), inSize, fp);
	fread(bic, sizeof(float), inSize, fp);
	this->preprocess();
	build = 1;
	return inSize;
}
int FrameData::preprocess(void)
{
	int i, temp;
#ifdef INT8x4_EXT_CONFIG
	xwtype*ppro_backup = ppro;
	for (i = 0;i < inSize;i++)
	{
		temp = bic[i] + 0.5;
		ppro[i] = temp - 128;
	}
	ppro = NCHW2NCHW_VECT_C_CPU(ppro, 1, 1, h, w, &outSize);
	free(ppro_backup);
#else
	for (i = 0;i < inSize;i++)
	{
		temp = bic[i] + 0.5;
		ppro[i] = (temp - 128.0) / 255.0;
	}
#endif
	return 0;
}
int FrameData::loadRes_GPU(convtype*v, float ratio_out)
{
	cudaDeviceSynchronize();
	cudaMemcpy(res, v, inSize*sizeof(convtype), cudaMemcpyDeviceToHost);
	this->ratio_out = ratio_out;
	return 0;
}
int FrameData::applyRes(void)
{
	int i;
	int temp;
	if (XWFORMAT == CUDNN_TENSOR_NCHW_VECT_C)
	{
		for (i = 0;i < inSize;i++)
		{
			temp = bic[i] + res[i] / (ratio_out/255) + 0.5;
			if (temp > 255)
				sr[i] = 255;
			else if (temp > 0)
				sr[i] = temp;
			else
				sr[i] = 0;
		}
	}
	else
	{
		for (i = 0;i < inSize;i++)
		{
			temp = bic[i] + res[i] * 255 + 0.5;
			if (temp > 255)
				sr[i] = 255;
			else if (temp > 0)
				sr[i] = temp;
			else
				sr[i] = 0;
		}
	}
	return 0;
}
double FrameData::count_psnr(void)
{
	int i, j;
	double mse, psnr;
	int shave = scale;
	mse = 0;
	for (i = shave; i < h - shave; i++)
		for (j = shave;j < w - shave;j++)
			mse += ((double)sr[i*w + j] - hr[i*w + j])*((double)sr[i*w + j] - hr[i*w + j]);
	mse /= (h - shave * 2)*(w - shave * 2);
	psnr = 10 * log10(65025.0 / mse);
	this->psnr_recon = psnr;
	mse = 0;
	for (i = shave; i < h - shave; i++)
		for (j = shave;j < w - shave;j++)
			mse += ((double)bic[i*w + j] - hr[i*w + j])*((double)bic[i*w + j] - hr[i*w + j]);
	mse /= (h - shave * 2)*(w - shave * 2);
	psnr = 10 * log10(65025.0 / mse);
	this->psnr_input = psnr;
	return psnr_recon;
}
int FrameData::save_recon_as(char* filename)
{
	FILE*fp;
	if (fopen_s(&fp, filename, "wb"))
	{
		printf("open file %s failed\n", filename);
		exit(1);
	}
	//sr = (datatype*)malloc(sizeof(datatype)*inSize);
	fwrite(sr, sizeof(datatype), inSize, fp);
	fclose(fp);
	return 0;
}
FrameData::~FrameData(void)
{
	if (build)
	{
		free(bic);
		free(hr);
		free(sr);
		free(ppro);
		free(res);
	}
}

yuv_data::yuv_data(const char*fn, int scale)
{
	int i;
	FILE *fp;
	frame = 0;
	this->scale = scale;
	if (fopen_s(&fp, fn, "rb"))
	{
		printf("open file %s failed\n", fn);
		exit(1);
	}
	while (data[frame].load(fp))
	{
		data[frame].scale = scale;
		frame++;
	}
}
double yuv_data::count_psnr(void)
{
	int i;
	double temp;
	temp = 0;
	for (i = 0;i < frame;i++)
		temp += data[i].psnr_input;
	temp /= frame;
	psnr_input = temp;
	temp = 0;
	for (i = 0;i < frame;i++)
		temp += data[i].psnr_recon;
	temp /= frame;
	psnr_recon = temp;
	//printf("psnr_input:%.2f,psnr_recon:%.2f\n", psnr_input, psnr_recon);
	return psnr_recon;
}
/*
int vrcnn_data::save_recon_as(char* filename)
{
	int i;
	FILE  *fp;
	if (fopen_s(&fp, filename, "wb"))
		printf("write file failed\n");
	datatype *uv = new datatype[h*w / 2];
	memset(uv, 0, h*w / 2);
	for (i = 0; i < frame; i++)
	{
		fwrite(recon + i * h * w, sizeof(datatype), h*w, fp);
		fwrite(uv, sizeof(datatype), h * w / 2, fp);
	}
	fclose(fp);
	return 0;
}*/
yuv_data::~yuv_data(void)
{
	return;
}
